"""
Offline Paraphraser: a desktop app for academic and legal writing.

Type or paste a sentence, click "Suggest 3 versions", and get three rewrites:
Corrected, Academic, and Concise. Everything runs on your own computer through
a local model (Ollama). No text is sent to the internet.

After each rewrite the app checks that numbers, citations, links and quotations
from your original are still there, and warns you if anything changed.

Standard library only (tkinter ships with Python on Windows).
"""
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections import Counter

import tkinter as tk
from tkinter import ttk, messagebox

OLLAMA = "http://127.0.0.1:11434"
RECOMMENDED_MODEL = "qwen2.5:3b"
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"

# ------------------------------------------------------------------ prompts

BASE_RULES = """You are an editing tool that rewrites one passage at a time.
Hard rules:
1. Keep the meaning exactly. Add no facts, examples, numbers, names or citations.
2. Copy these unchanged: every number, date, percentage, citation (for example [3] or (Smith, 2020)), footnote marker, URL, text inside quotation marks, proper name, defined term, and section, article or case reference.
3. Keep the strength of every claim. Do not turn "suggests" into "proves" or "may" into "will". Do not add hedging that was not in the original.
4. Write in the same language as the passage. Keep the same number of paragraphs.
5. Avoid filler and inflated words such as delve, pivotal, tapestry, underscore, landscape, "it is worth noting that".
6. Reply with only the rewritten passage. No preface, no notes, no quotation marks around it."""

REGISTER = {
    "Academic": "Register: formal academic prose. Neutral and precise. Passive voice and \"we\" are fine where natural.",
    "Legal scholarship": "Register: legal scholarship. Keep modal terms (shall, may, must, should) and terms of art exactly. Keep the order of reasoning and any signals such as 'see', 'cf.', 'but see'. Long sentences are acceptable if the structure stays clear.",
}

VARIANTS = [
    {
        "title": "Corrected",
        "hint": "Grammar, spelling and punctuation only. Your wording stays.",
        "instruction": "Task: fix grammar, spelling, punctuation, article and preposition errors only. Keep the author's wording and sentence structure wherever they are already correct. If there are no errors, return the passage unchanged.",
        "temp": 0.1,
        "min_ratio": 0.8,
        "default": True,
    },
    {
        "title": "Academic",
        "hint": "More formal and precise, with clearer flow.",
        "instruction": "Task: rewrite in formal, precise scholarly English. Improve clarity and flow and vary the sentence structure. Keep the order of ideas.",
        "temp": 0.4,
        "min_ratio": 0.6,
        "default": True,
    },
    {
        "title": "Concise",
        "hint": "Shorter and more direct.",
        "instruction": "Task: rewrite to be shorter and more direct. Remove redundancy and filler, split overlong sentences, and keep every claim and protected item.",
        "temp": 0.4,
        "min_ratio": 0.35,
        "default": False,
    },
    {
        "title": "Author voice",
        "hint": "Natural academic flow. Uses your writing sample if you add one.",
        "instruction": "Task: rewrite so it reads like a careful academic wrote it by hand: natural sentence rhythm with a mix of shorter and longer sentences, concrete wording, no stock phrases. Keep the order of ideas and every claim at its original strength.",
        "temp": 0.5,
        "min_ratio": 0.6,
        "uses_sample": True,
        "default": True,
    },
    {
        "title": "Active voice",
        "hint": "The doer is the subject: who does what.",
        "instruction": "Task: rewrite so that sentences use the active voice wherever the meaning allows, with the doer of the action as the grammatical subject. Where the doer is unknown, unstated or deliberately left out, keep that sentence as it is rather than inventing a doer. Keep modal terms exactly.",
        "temp": 0.2,
        "min_ratio": 0.6,
        "default": False,
    },
    {
        "title": "Passive voice",
        "hint": "The receiver is the subject: what is done.",
        "instruction": "Task: rewrite so that sentences use the passive voice wherever it is grammatical, with the receiver of the action as the grammatical subject. Keep the doer with 'by' if the original names it. Keep modal terms exactly.",
        "temp": 0.2,
        "min_ratio": 0.6,
        "default": False,
    },
]

# ----------------------------------------------------------------- verifier

NUM = re.compile(r"\d+(?:[.,]\d+)*")
BRACKET = re.compile(r"\[[^\[\]]{1,60}\]")
PAREN_YEAR = re.compile(r"\([^()]{0,80}\b(?:1[5-9]|20)\d{2}[a-z]?\b[^()]{0,40}\)")
URL = re.compile(r"https?://\S+")
QUOTE = re.compile(r"\u201c[^\u201d]{3,}\u201d|\"[^\"]{3,}\"")


def protected_items(text):
    # Numbers inside citations and links are reported under those headings instead.
    plain = URL.sub(" ", PAREN_YEAR.sub(" ", BRACKET.sub(" ", text)))
    return {
        "numbers": Counter(NUM.findall(plain)),
        "citations": Counter(BRACKET.findall(text) + PAREN_YEAR.findall(text)),
        "links": Counter(u.rstrip(".,;)") for u in URL.findall(text)),
        "quotations": Counter(QUOTE.findall(text)),
    }


def check(source, result, min_ratio=0.6):
    """Compare source and result. Returns a list of warning strings."""
    a, b = protected_items(source), protected_items(result)
    warnings = []
    for kind in a:
        missing = list((a[kind] - b[kind]).elements())
        added = list((b[kind] - a[kind]).elements())
        if missing:
            warnings.append("Missing " + kind + ": " + ", ".join(missing))
        if added:
            warnings.append("New " + kind + " not in your text: " + ", ".join(added))
    n_src, n_out = len(source.split()), len(result.split())
    if n_src >= 15:
        ratio = n_out / n_src
        if ratio < min_ratio:
            warnings.append("Much shorter than your text. Something may have been dropped.")
        elif ratio > 1.5:
            warnings.append("Much longer than your text. Something may have been added.")
    return warnings


# ------------------------------------------------------------------- ollama

def ollama(path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        OLLAMA + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ollama_up():
    try:
        ollama("/api/tags", timeout=2)
        return True
    except Exception:
        return False


def find_ollama():
    found = shutil.which("ollama")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidate = os.path.join(local, "Programs", "Ollama", "ollama.exe")
        if os.path.exists(candidate):
            return candidate
    return None


def list_models():
    tags = ollama("/api/tags", timeout=5)
    return [m["name"] for m in tags.get("models", [])]


def pull_model(name, on_progress):
    payload = json.dumps({"model": name, "name": name, "stream": True}).encode()
    req = urllib.request.Request(
        OLLAMA + "/api/pull", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                raise RuntimeError(d["error"])
            on_progress(d)


def clean(text):
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n|\n```$", "", text).strip()
    lines = text.split("\n")
    if len(lines) > 1 and re.match(r"^(here is|here's|sure|certainly)\b.*:$", lines[0].strip(), re.I):
        text = "\n".join(lines[1:]).strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


class Cancelled(Exception):
    pass


def stream_chat(payload, on_text=None):
    """Stream a chat reply. Returns (full_text, done_reason)."""
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    acc, reason = "", ""
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                raise RuntimeError(d["error"])
            acc += d.get("message", {}).get("content", "")
            if on_text:
                on_text(acc)
            if d.get("done"):
                reason = d.get("done_reason", "")
                break
    return acc, reason


def make_suggestion(text, variant, register, model, on_partial=None, sample=""):
    # Shared parts first, so Ollama can reuse its cache between versions.
    instruction = variant["instruction"]
    if variant.get("uses_sample") and sample.strip():
        instruction += ("\n\nStyle reference written by the author. Match its sentence length, vocabulary "
                        "and tone. Do not copy its content or facts:\n" + sample.strip()[:1500])
    system = "\n\n".join([BASE_RULES, REGISTER[register], instruction])
    payload = {
        "model": model,
        "stream": True,
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Passage:\n" + text},
        ],
        "options": {
            "temperature": variant["temp"],
            "num_ctx": 2048 if len(text) < 1500 else 4096,
            "num_predict": max(160, min(1200, int(len(text) * 0.9) + 80)),
        },
    }
    acc, reason = stream_chat(payload, on_partial)
    out = clean(acc)
    warnings = check(text, out, variant["min_ratio"])
    if reason == "length":
        warnings.append("The answer was cut off. Try a shorter passage.")
    return out, warnings


def friendly(e):
    if isinstance(e, urllib.error.HTTPError):
        if e.code == 404:
            return "Model not found. Choose an installed model, or click Download."
        try:
            return "Ollama error: " + e.read().decode("utf-8", "replace")[:200]
        except Exception:
            return "Ollama error (HTTP %d)." % e.code
    if isinstance(e, urllib.error.URLError):
        return "Cannot reach Ollama. Is it running?"
    return str(e)


# ---------------------------------------------------------------------- GUI

LIGHT = {
    "BG": "#f3f5f7", "PANEL": "#ffffff", "INK": "#17212b", "MUTED": "#5b6875", "LINE": "#d5dbe1",
    "ACCENT": "#17615a", "ACCENT_HOVER": "#124d47", "ACCENT_DIS": "#9db8b4",
    "WARN_BG": "#fff1e0", "WARN_INK": "#8a4300", "ERR": "#a12a2a",
    "BTN": "#e4e8ec", "BTN_HOVER": "#d3d9df",
}
DARK = {
    "BG": "#12171d", "PANEL": "#1a2129", "INK": "#e6ebf0", "MUTED": "#98a5b2", "LINE": "#2c3742",
    "ACCENT": "#2f8f83", "ACCENT_HOVER": "#3aa396", "ACCENT_DIS": "#2a4a46",
    "WARN_BG": "#33240f", "WARN_INK": "#f0b46a", "ERR": "#ef8a8a",
    "BTN": "#27313c", "BTN_HOVER": "#34414f",
}
BG = PANEL = INK = MUTED = LINE = ACCENT = ACCENT_HOVER = ACCENT_DIS = ""
WARN_BG = WARN_INK = ERR = BTN = BTN_HOVER = ""


def set_palette(p):
    globals().update(p)


set_palette(LIGHT)

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".offline_paraphraser.json")


def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(data):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def system_is_dark():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except Exception:
        return False


COLOR_OPTIONS = ("bg", "fg", "highlightbackground", "highlightcolor", "activebackground",
                 "activeforeground", "insertbackground", "selectcolor", "disabledforeground")


def recolor(widget, mapping):
    """Swap palette colors on a widget and all its children."""
    for opt in COLOR_OPTIONS:
        try:
            value = str(widget.cget(opt)).lower()
        except tk.TclError:
            continue
        if value in mapping:
            try:
                widget.configure(**{opt: mapping[value]})
            except tk.TclError:
                pass
    for child in widget.winfo_children():
        recolor(child, mapping)


UI = "Segoe UI"
SERIF = ("Georgia", 12)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        name = self.settings.get("theme", "System")
        dark = system_is_dark() if name == "System" else name == "Dark"
        set_palette(DARK if dark else LIGHT)
        self.title("Offline Paraphraser")
        self.geometry("1000x700")
        self.minsize(760, 560)
        self.configure(bg=BG)
        self.q = queue.Queue()
        self.run_id = 0
        self.busy = False
        self.cards = []
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self.after(100, self._poll)
        threading.Thread(target=self._startup, daemon=True).start()

    # -- layout ---------------------------------------------------------
    def _apply_styles(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure("TButton", font=(UI, 10), padding=(10, 5), background=BTN, foreground=INK,
                     bordercolor=LINE, focuscolor=BTN, lightcolor=BTN, darkcolor=BTN)
        st.map("TButton", background=[("active", BTN_HOVER), ("disabled", BTN)],
               foreground=[("disabled", MUTED)])
        st.configure("Accent.TButton", background=ACCENT, foreground="white",
                     font=(UI, 10, "bold"), bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT)
        st.map("Accent.TButton", background=[("active", ACCENT_HOVER), ("disabled", ACCENT_DIS)],
               foreground=[("disabled", "white")])
        st.configure("TCombobox", padding=4, fieldbackground=PANEL, background=BTN, foreground=INK,
                     arrowcolor=INK, bordercolor=LINE, lightcolor=BTN, darkcolor=BTN)
        st.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", INK)],
               background=[("readonly", BTN)])
        st.configure("Vertical.TScrollbar", background=BTN, troughcolor=BG, arrowcolor=INK, bordercolor=BG)
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", INK)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", "white")

    def _build(self):
        self._apply_styles()

        root = tk.Frame(self, bg=BG, padx=16, pady=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)

        top = tk.Frame(root, bg=BG)
        top.grid(row=0, column=0, sticky="ew")
        tk.Label(top, text="Model", bg=BG, fg=MUTED, font=(UI, 9)).pack(side="left")
        self.model = tk.StringVar(value=self.settings.get("model", ""))
        self.model_box = ttk.Combobox(top, textvariable=self.model, width=22)
        self.model_box.pack(side="left", padx=(6, 4))
        ttk.Button(top, text="Download", command=self.download).pack(side="left", padx=(0, 16))
        tk.Label(top, text="Style", bg=BG, fg=MUTED, font=(UI, 9)).pack(side="left")
        saved_reg = self.settings.get("register", "Academic")
        self.register = tk.StringVar(value=saved_reg if saved_reg in REGISTER else "Academic")
        ttk.Combobox(top, textvariable=self.register, values=list(REGISTER),
                     state="readonly", width=16).pack(side="left", padx=6)
        tk.Label(top, text="Theme", bg=BG, fg=MUTED, font=(UI, 9)).pack(side="left", padx=(10, 0))
        self.theme = tk.StringVar(value=self.settings.get("theme", "System"))
        theme_box = ttk.Combobox(top, textvariable=self.theme, values=["System", "Light", "Dark"],
                                 state="readonly", width=7)
        theme_box.pack(side="left", padx=6)
        theme_box.bind("<<ComboboxSelected>>", lambda e: self.apply_theme())

        tk.Label(root, text="Your sentence or paragraph", bg=BG, fg=INK,
                 font=(UI, 10, "bold")).grid(row=1, column=0, sticky="w", pady=(10, 4))
        box = tk.Frame(root, bg=LINE, padx=1, pady=1)
        box.grid(row=2, column=0, sticky="ew")
        self.inp = tk.Text(box, height=5, wrap="word", font=SERIF, bd=0, bg=PANEL, fg=INK,
                           padx=10, pady=8, undo=True, insertbackground=INK,
                           highlightthickness=0)
        self.inp.pack(fill="both", expand=True)
        self.inp.bind("<Control-Return>", lambda e: (self.suggest(), "break")[1])
        self.inp.bind("<KeyRelease>", lambda e: self._count())
        self.inp.focus_set()

        bar = tk.Frame(root, bg=BG)
        bar.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.go = ttk.Button(bar, text="Suggest versions", style="Accent.TButton", command=self.suggest)
        self.go.pack(side="left")
        ttk.Button(bar, text="Clear", command=self.clear).pack(side="left", padx=8)
        ttk.Button(bar, text="Writing sample...", command=self.open_sample).pack(side="left")
        self.count = tk.Label(bar, text="0 words  (Ctrl+Enter to run)", bg=BG, fg=MUTED, font=(UI, 9))
        self.count.pack(side="left", padx=8)
        self.status = tk.Label(bar, text="Starting...", bg=BG, fg=MUTED, font=(UI, 9), anchor="e")
        self.status.pack(side="right")

        # scrollable results area
        wrap = tk.Frame(root, bg=BG)
        wrap.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        root.rowconfigure(4, weight=1)
        self.canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=BG)
        win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self._resize(win, e.width))
        self.bind_all("<MouseWheel>", self._wheel)

        for i, v in enumerate(VARIANTS):
            card = tk.Frame(self.inner, bg=PANEL, highlightbackground=LINE, highlightthickness=1,
                            padx=12, pady=8)
            card.pack(fill="x", pady=(0, 8))
            head = tk.Frame(card, bg=PANEL)
            head.pack(fill="x")
            ticks = self.settings.get("ticks", {})
            var = tk.BooleanVar(value=bool(ticks.get(v["title"], v.get("default", True))))
            tk.Checkbutton(head, text=v["title"], variable=var, bg=PANEL, fg=INK,
                           activebackground=PANEL, activeforeground=INK, selectcolor=PANEL,
                           font=(UI, 10, "bold"), anchor="w",
                           highlightthickness=0).pack(side="left")
            tk.Label(head, text=v["hint"], bg=PANEL, fg=MUTED, font=(UI, 9)).pack(side="left", padx=8)
            ttk.Button(head, text="Copy", command=lambda i=i: self.copy(i)).pack(side="right")
            ttk.Button(head, text="Use this", command=lambda i=i: self.use(i)).pack(side="right", padx=6)
            txt = tk.Text(card, height=2, wrap="word", font=SERIF, bd=0, bg=PANEL, fg=INK,
                          padx=0, pady=6, relief="flat", state="disabled",
                          highlightthickness=0)
            txt.pack(fill="x")
            warn = tk.Label(card, text="", bg=WARN_BG, fg=WARN_INK, font=(UI, 9),
                            justify="left", anchor="w", wraplength=860, padx=8, pady=4)
            self.cards.append({"text": txt, "warn": warn, "var": var})

    def _resize(self, win, width):
        self.canvas.itemconfigure(win, width=width)
        for c in self.cards:
            c["warn"].config(wraplength=max(300, width - 70))

    def _wheel(self, e):
        if e.widget is self.inp:
            return
        self.canvas.yview_scroll(int(-e.delta / 120) or (-1 if e.delta > 0 else 1), "units")

    # -- helpers --------------------------------------------------------
    def post(self, kind, *args):
        self.q.put((kind, args))

    def set_status(self, text, bad=False):
        self.status.config(text=text, fg=ERR if bad else MUTED)

    def _fit(self, txt):
        try:
            txt.update_idletasks()
            n = txt.count("1.0", "end-1c", "displaylines")
            n = n[0] if n else 1
        except Exception:
            n = 3
        txt.config(height=max(2, min(n + 1, 16)))

    def _set_card(self, i, text, warnings=(), error=None):
        c = self.cards[i]
        c["text"].config(state="normal", fg=ERR if error else INK)
        c["text"].delete("1.0", "end")
        c["text"].insert("1.0", error or text)
        c["text"].config(state="disabled")
        self._fit(c["text"])
        shown = [] if error else list(warnings)
        if shown:
            c["warn"].config(text="Check: " + "  |  ".join(shown))
            c["warn"].pack(fill="x", pady=(4, 0))
        else:
            c["warn"].pack_forget()

    def _card_text(self, i):
        return self.cards[i]["text"].get("1.0", "end").strip()

    def _count(self):
        n = len(self.inp.get("1.0", "end").split())
        self.count.config(text="%d words  (Ctrl+Enter to run)" % n)

    # -- actions --------------------------------------------------------
    def suggest(self):
        if self.busy:
            return
        text = self.inp.get("1.0", "end").strip()
        if not text:
            return
        picks = [i for i, c in enumerate(self.cards) if c["var"].get()]
        if not picks:
            messagebox.showinfo("Nothing selected", "Tick at least one version.")
            return
        model = self.model.get().strip()
        if not model:
            messagebox.showinfo("No model", "Type a model name and click Download, "
                                "or choose an installed one.\nSuggested: " + RECOMMENDED_MODEL)
            return
        self.run_id += 1
        self.busy = True
        self.go.config(state="disabled")
        for i in range(len(VARIANTS)):
            self._set_card(i, "Waiting..." if i in picks else "")
        threading.Thread(target=self._work,
                         args=(self.run_id, text, model, self.register.get(), picks,
                               self.settings.get("sample", "")),
                         daemon=True).start()

    def _work(self, rid, text, model, register, picks, sample=""):
        for n, i in enumerate(picks):
            if rid != self.run_id:
                return
            self.post("working", rid, i, n + 1, len(picks))
            last = [0.0]

            def partial(acc):
                if rid != self.run_id:
                    raise Cancelled()
                now = time.time()
                if now - last[0] > 0.15:
                    last[0] = now
                    self.post("partial", rid, i, acc)
            try:
                out, warnings = make_suggestion(text, VARIANTS[i], register, model, partial, sample)
                self.post("result", rid, i, out, warnings)
            except Cancelled:
                return
            except Exception as e:
                self.post("error", rid, i, friendly(e))
        self.post("done", rid)

    def clear(self):
        self.run_id += 1
        self.busy = False
        self.go.config(state="normal")
        self.inp.delete("1.0", "end")
        for i in range(len(VARIANTS)):
            self._set_card(i, "")
        self._count()
        self.inp.focus_set()

    def copy(self, i):
        t = self._card_text(i)
        if t:
            self.clipboard_clear()
            self.clipboard_append(t)
            self.set_status("Copied.")

    def use(self, i):
        t = self._card_text(i)
        if t:
            self.inp.delete("1.0", "end")
            self.inp.insert("1.0", t)
            self._count()

    def download(self):
        name = self.model.get().strip() or RECOMMENDED_MODEL
        self.model.set(name)
        if not ollama_up():
            messagebox.showinfo("Ollama not running", "Ollama is not running yet. Wait a moment, or install it from "
                                + OLLAMA_DOWNLOAD_URL)
            return
        if not messagebox.askyesno("Download model",
                                   "Download '%s'? This needs internet once and can be a few GB.\n"
                                   "After that the app works offline." % name):
            return
        threading.Thread(target=self._pull, args=(name,), daemon=True).start()

    def _pull(self, name):
        def progress(d):
            total, done = d.get("total"), d.get("completed")
            if total and done is not None:
                self.post("status", "Downloading %s: %d%%" % (name, done * 100 // total), False)
            else:
                self.post("status", "%s: %s" % (name, d.get("status", "")), False)
        try:
            pull_model(name, progress)
            self.post("models", list_models(), name)
            self._warm(name)
        except Exception as e:
            self.post("status", "Download failed: " + friendly(e), True)

    def _warm(self, model):
        """Load the model into memory now, so the first suggestion starts faster."""
        self.post("status", "Loading model into memory...", False)
        try:
            ollama("/api/generate", {"model": model, "prompt": "", "keep_alive": "30m"}, timeout=300)
            self.post("status", "Ready. Everything runs on this computer.", False)
        except Exception:
            self.post("status", "Ready.", False)

    def apply_theme(self):
        name = self.theme.get()
        dark = system_is_dark() if name == "System" else name == "Dark"
        new = DARK if dark else LIGHT
        mapping = {globals()[k].lower(): new[k] for k in new if globals()[k] != new[k]}
        if mapping:
            recolor(self, mapping)
            set_palette(new)
            self._apply_styles()
        self.settings["theme"] = name
        save_settings(self.settings)

    def open_sample(self):
        win = tk.Toplevel(self)
        win.title("Your writing sample")
        win.configure(bg=BG)
        win.geometry("640x430")
        win.transient(self)
        tk.Label(win, text="Paste 100 to 300 words that you wrote yourself. The Author voice version "
                           "will imitate its sentence length, vocabulary and tone. It is stored only "
                           "on this computer.", wraplength=600, justify="left", bg=BG, fg=INK,
                 font=(UI, 10)).pack(anchor="w", padx=14, pady=(12, 6))
        box = tk.Text(win, wrap="word", font=SERIF, bd=0, bg=PANEL, fg=INK, insertbackground=INK,
                      padx=10, pady=8, highlightthickness=1, highlightbackground=LINE)
        box.pack(fill="both", expand=True, padx=14)
        box.insert("1.0", self.settings.get("sample", ""))
        row = tk.Frame(win, bg=BG)
        row.pack(fill="x", padx=14, pady=10)

        def save():
            self.settings["sample"] = box.get("1.0", "end").strip()[:3000]
            save_settings(self.settings)
            win.destroy()
            self.set_status("Writing sample saved." if self.settings["sample"] else "Writing sample removed.")
        ttk.Button(row, text="Save", style="Accent.TButton", command=save).pack(side="right")
        ttk.Button(row, text="Cancel", command=win.destroy).pack(side="right", padx=8)
        box.focus_set()

    def _close(self):
        self.settings.update({
            "theme": self.theme.get(),
            "model": self.model.get().strip(),
            "register": self.register.get(),
            "ticks": {v["title"]: c["var"].get() for v, c in zip(VARIANTS, self.cards)},
        })
        save_settings(self.settings)
        self.destroy()

    # -- startup (worker thread) ----------------------------------------
    def _startup(self):
        self.post("status", "Checking Ollama...", False)
        if not ollama_up():
            exe = find_ollama()
            if not exe:
                self.post("no_ollama")
                return
            self.post("status", "Starting Ollama...", False)
            try:
                subprocess.Popen([exe, "serve"], creationflags=NO_WINDOW,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                self.post("status", "Could not start Ollama: %s" % e, True)
                return
            for _ in range(40):
                time.sleep(0.5)
                if ollama_up():
                    break
            else:
                self.post("status", "Ollama did not start. Open it from the Start menu.", True)
                return
        try:
            names = list_models()
            self.post("models", names, None)
            if names:
                saved = load_settings().get("model")
                self._warm(saved if saved in names else names[0])
        except Exception as e:
            self.post("status", friendly(e), True)

    # -- queue pump (main thread) ---------------------------------------
    def _poll(self):
        try:
            while True:
                kind, a = self.q.get_nowait()
                if kind == "status":
                    self.set_status(a[0], a[1])
                elif kind == "models":
                    names, select = a
                    self.model_box["values"] = names
                    if select:
                        self.model.set(select)
                    elif names and self.model.get() not in names:
                        self.model.set(names[0])
                    if not names:
                        self.model.set(self.model.get() or RECOMMENDED_MODEL)
                        self.set_status("No model yet. Click Download (one time, needs internet).", True)
                elif kind == "no_ollama":
                    self.set_status("Ollama is not installed.", True)
                    if messagebox.askyesno("Ollama needed",
                                           "This app uses Ollama to run the language model on your computer.\n\n"
                                           "Open the download page now? Install it, then start this app again."):
                        webbrowser.open(OLLAMA_DOWNLOAD_URL)
                elif a and a[0] == self.run_id:
                    if kind == "working":
                        self._set_card(a[1], "Working...")
                        self.set_status("Writing version %d of %d..." % (a[2], a[3]))
                    elif kind == "partial":
                        self._set_card(a[1], a[2])
                    elif kind == "result":
                        self._set_card(a[1], a[2], a[3])
                    elif kind == "error":
                        self._set_card(a[1], "", error=a[2])
                    elif kind == "done":
                        self.busy = False
                        self.go.config(state="normal")
                        self.set_status("Done. Read each version before you use it.")
        except queue.Empty:
            pass
        self.after(100, self._poll)


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()


if __name__ == "__main__":
    main()
