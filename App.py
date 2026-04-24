import re
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from typing import List, Dict, Set, Tuple

from transformers import pipeline

try:
    import pypdf
    PYPDF_AVAILABLE = True
except Exception:
    PYPDF_AVAILABLE = False


#  Model 
# cross-encoder/nli-deberta-v3-small is a pretrained NLI (Natural Language
# Inference) classifier from Hugging Face.
# The model reads both the input line and the candidate label together and
# outputs an entailment probability — reasoning about how well the line matches.
MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Minimum entailment probability to accept a classification.
CONFIDENCE_THRESHOLD = 0.40


#  Bucket definitions 
# Buckets are pre-defined categories we want to sort rubric lines into.
# Each bucket has a human-friendly label, a display name for the UI, and a color.
# Label strings get passed directly to the NLI model as candidates.
# Phrasing them as hypothesis statements gets better entailment scores.
BUCKETS = {
    "main_task": {
        "label": "the primary deliverable: write, build, create, or implement a program or application",
        "display": "Main Task",
        "color": "#1a56db",
    },
    "output_requirement": {
        "label": "a specific requirement about how the program output must be formatted, sorted, or displayed",
        "display": "Output Requirement",
        "color": "#7e3af2",
    },
    "supported_option": {
        "label": "a command-line flag, option, or argument the program must accept or support",
        "display": "Supported Options / Features",
        "color": "#0694a2",
    },
    "allowed_language": {
        "label": "which programming language or languages are allowed or required for this assignment",
        "display": "Allowed Languages",
        "color": "#057a55",
    },
    "exception_note": {
        "label": "something the student does not need to do, an exception, or an optional feature explicitly excluded",
        "display": "Exceptions / Notes",
        "color": "#d97706",
    },
    "submission_requirement": {
        "label": "when or how to submit the assignment: deadline, due date, upload instructions, or late penalty",
        "display": "Submission",
        "color": "#e02424",
    },
}

_BUCKET_KEYS = list(BUCKETS.keys())
_CANDIDATE_LABELS = [BUCKETS[k]["label"] for k in _BUCKET_KEYS]


#  Noise patterns — lines matching these are always dropped before NLI 
# These are lines that look actionable (contain "must", "should", etc.) but are
# actually just filler, formatting, or metadata in typical rubrics/PDFs.
_NOISE_PATTERNS = [
    r"^\s*page\s+\d+",                          # "Page 1", "Page 2 of 5"
    r"^\s*\d+\s*$",                              # bare page numbers
    r"^\s*course\s*:",                           # "Course: CS101"
    r"^\s*instructor\s*:",                       # "Instructor: Dr. Smith"
    # NOTE: "Due Date:" is NOT noise — it is a submission cue, so we keep it
    r"^\s*assignment\s+\d+\s*$",                 # "Assignment 1"
    r"^\s*total\s+points?\s*:",                  # "Total Points: 100"
    r"^\s*grading\s+criteria\s*$",               # bare section headers
    r"^\s*rubric\s*$",
    r"^\s*instructions?\s*$",
    r"^[-=_*]{3,}$",                             # divider lines --- === ***
    r"^\s*https?://\S+\s*$",                     # bare URLs
    r"^\s*[(\[{]?\d+[)\]}]?\s*points?\s*$",      # "(10 points)" standalone
]

#  Strong actionable patterns — lines matching these always pass the filter 
# These are highly specific to assignment instructions and very rarely appear
# in filler text, so we trust them without requiring NLI.
_STRONG_PATTERNS = [
    r"\bwrite\s+a\b", r"\bwrite\s+an\b",         # "write a program"
    r"\bimplement\b", r"\bdevelop\b",
    r"\bcreate\s+a\b", r"\bbuild\s+a\b",
    r"\bmust\s+\w+\b",                            # "must handle", "must support"
    r"\bshould\s+\w+\b",
    r"\bdo\s+not\s+(have\s+to|need\s+to)\b",      # explicit exceptions
    r"\bnot\s+required\b", r"\bnot\s+necessary\b",
    r"\bsubmit\b", r"\bupload\b", r"\bturn\s+in\b",
    r"\bdeadline\b", r"\bdue\s+(date|by|on|friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b",
    r"\blate\s+submission\b", r"\blate\s+penalty\b",
    r"\bdue\b", r"\bby\s+(friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b",
    r"\b\d{1,2}[:/]\d{2}\s*(am|pm)\b",           # times like 11:59pm
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",  # month names
    r"\bcanvas\b", r"\bgradescope\b", r"\bblackboard\b",  # LMS names
    r"\bzip\s+file\b", r"\b\.pdf\b", r"\b\.zip\b",     # file types

    r"\bhandle\s+\w+\b",                          # "handle the case"
    r"\bsupport\s+\w+\b",                         # "support these options"
    r"\bthe\s+output\s+(must|should|will)\b",
    r"\bwork\s+exactly\b", r"\bidentical\b",
    r"\ballowed\s+(language|to\s+use)\b",
    r"\buse\s+(any|one\s+of|the\s+following)\b",
    r"\bnote\s*:\b",
]

#  Weak actionable patterns — lines need at least 2 to pass 
# Words that often appear in actionable lines but also in filler text.
# Requiring 2+ reduces false positives.
_WEAK_PATTERNS = [
    r"\buse\b", r"\binclude\b", r"\bdisplay\b",
    r"\boutput\b", r"\bproduce\b", r"\ballow\b",
    r"\brequire\b", r"\bhandle\b", r"\bnote\b",
    r"\bby\b", r"\bdate\b", r"\bprogram\b",
    r"\bfile\b", r"\bformat\b", r"\bread\b",
    r"\bprint\b", r"\bshow\b", r"\bcheck\b",
]


class RubricHelperApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Rubric Helper")
        self.root.geometry("1100x780")
        self.root.configure(bg="#f8f8f8")

        self.classifier = None
        self.model_ready = False
        self.placeholder_text = "Paste assignment prompt/rubric here, or click Import PDF..."
        self.input_placeholder_active = False

        self._build_gui()
        threading.Thread(target=self._load_model, daemon=True).start()

    #  GUI 

    def _build_gui(self):
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except Exception:
            style.theme_use("clam")

        bg = "#f3f6fb"
        panel_bg = "#ffffff"
        text_main = "#0f172a"
        text_muted = "#475569"
        border = "#dbe3ef"
        accent = "#2563eb"

        self.root.configure(bg=bg)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=text_main)
        style.configure("Card.TFrame", background=panel_bg, relief="solid", borderwidth=1)
        style.configure("Legend.TFrame", background=panel_bg, relief="flat", borderwidth=0)
        style.configure("Bold.TLabel", background=bg, foreground=text_main, font=("Segoe UI", 11, "bold"))
        style.configure("Title.TLabel", background=bg, foreground=text_main, font=("Segoe UI", 20, "bold"))
        style.configure("Sub.TLabel", background=bg, foreground=text_muted, font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background=panel_bg, foreground=text_main, font=("Segoe UI", 11, "bold"))
        style.configure("CardSub.TLabel", background=panel_bg, foreground=text_muted, font=("Segoe UI", 10))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure("Progress.Horizontal.TProgressbar", troughcolor="#e2e8f0", background=accent, bordercolor="#e2e8f0")

        main = ttk.Frame(self.root, padding=18)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Rubric Helper", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            main,
            text=f"Zero-shot NLI classification via {MODEL_NAME}",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 14))

        paned = ttk.Panedwindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True)

        left_panel = ttk.Frame(paned, padding=16, style="Card.TFrame")
        right_panel = ttk.Frame(paned, padding=16, style="Card.TFrame")
        paned.add(left_panel, weight=1)
        paned.add(right_panel, weight=1)

        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(2, weight=1)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(2, weight=1)

        #  Input (grid so text box lines up with results box)
        ttk.Label(left_panel, text="Assignment Prompt / Rubric", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            left_panel,
            text="Paste text directly, or import a PDF with selectable text.",
            style="CardSub.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))
        self.input_text = scrolledtext.ScrolledText(
            left_panel,
            wrap=tk.WORD,
            height=20,
            font=("Segoe UI", 11),
            relief="flat",
            borderwidth=1,
            bg="#fbfdff",
            fg=text_main,
            insertbackground=text_main,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=accent,
            padx=10,
            pady=10,
        )
        self.input_text.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        self.input_text.bind("<FocusIn>", self._clear_placeholder)
        self.input_text.bind("<FocusOut>", self._restore_placeholder_if_empty)
        self._set_placeholder()

        #  Buttons
        btn_row = ttk.Frame(left_panel)
        btn_row.grid(row=3, column=0, sticky="ew")

        self.analyze_btn = ttk.Button(
            btn_row, text="Analyze Requirements", command=self._on_analyze, style="Primary.TButton"
        )
        self.analyze_btn.pack(side="left")
        self.analyze_btn.state(["disabled"])

        self.pdf_btn = ttk.Button(btn_row, text="Import PDF", command=self._upload_pdf)
        self.pdf_btn.pack(side="left", padx=8)

        ttk.Button(btn_row, text="Clear All", command=self._clear).pack(side="left")

        #  Results (mirror left column: title, helper row, text, buttons)
        ttk.Label(right_panel, text="Extracted Requirements", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        legend = ttk.Frame(right_panel, style="Legend.TFrame")
        legend.grid(row=1, column=0, sticky="w", pady=(0, 8))
        for key, meta in BUCKETS.items():
            tk.Label(
                legend,
                text="●",
                fg=meta["color"],
                bg=panel_bg,
                font=("Segoe UI", 12),
                bd=0,
                highlightthickness=0,
                relief="flat",
            ).pack(side="left")
            tk.Label(
                legend,
                text=meta["display"] + "  ",
                bg=panel_bg,
                fg=text_muted,
                font=("Segoe UI", 9),
                bd=0,
                highlightthickness=0,
                relief="flat",
            ).pack(side="left")

        self.results_text = scrolledtext.ScrolledText(
            right_panel,
            wrap=tk.WORD,
            height=20,
            font=("Consolas", 10),
            relief="flat",
            borderwidth=1,
            bg="#f8fafc",
            fg=text_main,
            insertbackground=text_main,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=accent,
            padx=10,
            pady=10,
            state="disabled",
        )
        self.results_text.grid(row=2, column=0, sticky="nsew", pady=(0, 8))

        results_btn_row = ttk.Frame(right_panel)
        results_btn_row.grid(row=3, column=0, sticky="ew")
        ttk.Button(results_btn_row, text="Copy Text", command=self._copy_results).pack(side="left")
        ttk.Button(results_btn_row, text="Save File", command=self._save_results).pack(side="left", padx=8)

        # Shared footer so both panels keep the same vertical space for text boxes
        footer = ttk.Frame(main)
        footer.pack(fill="x", pady=(10, 0))
        self.progress = ttk.Progressbar(
            footer,
            mode="indeterminate",
            style="Progress.Horizontal.TProgressbar",
            length=220,
        )
        self.progress.pack(fill="x", pady=(0, 6))
        self.progress.start(10)

        ttk.Label(
            footer,
            text="Tip: cleaner rubric text usually gives better classification results.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        self.status_var = tk.StringVar(value="Loading NLI model…")
        ttk.Label(footer, textvariable=self.status_var, style="Sub.TLabel").pack(anchor="w")

        for key, meta in BUCKETS.items():
            self.results_text.tag_config(f"header_{key}", foreground=meta["color"],
                                         font=("Segoe UI", 11, "bold"))
            self.results_text.tag_config(f"item_{key}", foreground=text_main,
                                         font=("Consolas", 10))
            self.results_text.tag_config(f"conf_{key}", foreground=meta["color"],
                                         font=("Consolas", 9))

        self.results_text.tag_config("unclassified_header", foreground="#888",
                                     font=("Segoe UI", 11, "bold"))
        self.results_text.tag_config("unclassified_item", foreground="#888",
                                     font=("Consolas", 10))
        self.results_text.tag_config("loading", foreground="#555", font=("Segoe UI", 11, "italic"))
        self.results_text.tag_config("rule", foreground="#ccc")

    #  PDF Upload 

    def _upload_pdf(self):
        """Open a file dialog, extract text from the chosen PDF, and load it into the input box."""
        path = filedialog.askopenfilename(
            title="Select a PDF rubric",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            text = self._extract_pdf_text(path)
            if not text.strip():
                messagebox.showwarning("Empty PDF", "No readable text found in this PDF. It may be scanned/image-based.")
                return
            self.input_placeholder_active = False
            self.input_text.delete("1.0", tk.END)
            self.input_text.configure(fg="#0f172a")
            self.input_text.insert("1.0", text)
            self._set_status(f"✓ PDF loaded — {len(text.split())} words extracted")
        except Exception as exc:
            err = str(exc)
            messagebox.showerror("PDF Error", f"Could not read PDF:\n{err}")

    @staticmethod
    def _extract_pdf_text(path: str) -> str:
        """
        Extract plain text from a PDF using pypdf, then clean it up.
        pypdf often returns text with no line breaks or single characters
        on each line depending on the PDF encoding. This method reconstructs
        readable lines from whatever pypdf gives back.
        """
        if not PYPDF_AVAILABLE:
            raise RuntimeError(
                "pypdf is not installed. Run: pip install pypdf"
            )

        reader = pypdf.PdfReader(path)
        pages = []
        for page in reader.pages:
            # Use layout extraction mode for better line detection
            page_text = page.extract_text(extraction_mode="layout")
            if not page_text:
                # fallback to default mode
                page_text = page.extract_text()
            if page_text:
                pages.append(page_text)

        raw = "\n".join(pages)
        return RubricHelperApp._clean_pdf_text(raw)

    @staticmethod
    def _clean_pdf_text(text: str) -> str:
        """
        Reconstruct clean, consistent lines from messy PDF-extracted text.

        Fixes:
          1. Word-per-line: words extracted one per line, rejoins into sentences
          2. Run-on text: no line breaks at all, splits on sentence boundaries
          3. Extra whitespace, leading symbols, blank lines
          4. Normalizes spacing so PDF and pasted text produce identical output
        """
        lines = text.split("\n")

        # Step 1  strip leading/trailing whitespace and symbols from each line
        lines = [l.strip(" \t\r-•*|") for l in lines]

        # Step 2  detect word-per-line mode
        non_empty = [l for l in lines if l]
        single_word = [l for l in non_empty if len(l.split()) == 1]

        if non_empty and (len(single_word) / len(non_empty)) > 0.5:
            # Rejoin everything, then split on sentence-ending punctuation
            rejoined = " ".join(non_empty)
            # Normalize multiple spaces
            rejoined = re.sub(r' {2,}', ' ', rejoined)
            sentences = re.split(r'(?<=[.!?:])\s+(?=[A-Z0-9/])', rejoined)
            lines = [s.strip() for s in sentences if s.strip()]
        else:
            # Step 3  merge continuation lines if a line doesn't end with
            # punctuation and the next line starts lowercase, they're one sentence
            merged = []
            buffer = ""
            for line in non_empty:
                if not line:
                    continue
                if buffer:
                    # If buffer doesn't end with sentence-ending punctuation
                    # and this line starts lowercase, merge
                    if not re.search(r'[.!?:]$', buffer) and line[0].islower():
                        buffer = buffer + " " + line
                    else:
                        merged.append(buffer)
                        buffer = line
                else:
                    buffer = line
            if buffer:
                merged.append(buffer)
            lines = merged

        # Step 4  final cleanup normalize whitespace within each line
        cleaned = []
        for line in lines:
            line = re.sub(r' {2,}', ' ', line).strip()
            if len(line.split()) >= 2:   # drop single-word remnants
                cleaned.append(line)

        return "\n".join(cleaned)

    #  Model loading 

    def _load_model(self):
        try:
            self._set_status("Downloading model weights… (first run only, ~180MB)")
            # pipeline zero-shot-classification wraps the NLI model so we can
            # pass any text + candidate labels and get back entailment probabilities.
            self.classifier = pipeline(
                "zero-shot-classification",
                model=MODEL_NAME,
                device=-1,  # CPU change to 0 for GPU
            )
            self.model_ready = True
            self.root.after(0, self._on_model_ready)
        except Exception as exc:
            err_msg = str(exc)
            self.root.after(0, lambda: messagebox.showerror("Model Error", err_msg))

    def _on_model_ready(self):
        self.analyze_btn.state(["!disabled"])
        self.progress.stop()
        self._set_status(f"✓ {MODEL_NAME} ready")

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _set_placeholder(self):
        if self.input_text.get("1.0", tk.END).strip():
            return
        self.input_placeholder_active = True
        self.input_text.insert("1.0", self.placeholder_text)
        self.input_text.configure(fg="#64748b")

    def _clear_placeholder(self, _event=None):
        if not self.input_placeholder_active:
            return
        self.input_text.delete("1.0", tk.END)
        self.input_text.configure(fg="#0f172a")
        self.input_placeholder_active = False

    def _restore_placeholder_if_empty(self, _event=None):
        if self.input_text.get("1.0", tk.END).strip():
            return
        self.input_text.delete("1.0", tk.END)
        self._set_placeholder()

    # NLI Classification 

    def _classify(self, line: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Run zero-shot NLI classification on a single line.

        The model receives:
            premise    = the rubric line
            hypothesis = "This sentence is about [candidate label]"

        It returns entailment scores for each candidate label.
        The highest score wins, unless it is below CONFIDENCE_THRESHOLD.

        Returns:
            (bucket_key, top_score, {bucket_key: score, ...})
        """
        result = self.classifier(
            line,
            candidate_labels=_CANDIDATE_LABELS,
            multi_label=False,
        )

        all_scores: Dict[str, float] = {}
        for label, score in zip(result["labels"], result["scores"]):
            idx = _CANDIDATE_LABELS.index(label)
            all_scores[_BUCKET_KEYS[idx]] = score

        best_key = max(all_scores, key=all_scores.__getitem__)
        best_score = all_scores[best_key]

        if best_score < CONFIDENCE_THRESHOLD:
            return "unclassified", best_score, all_scores

        return best_key, best_score, all_scores

    # Text processings

    @staticmethod
    def _split_lines(text: str) -> List[str]:
        """
        Split text into lines. Handles both newline-separated rubrics and
        sentence-separated paragraphs (common in PDF-extracted text).
        Short fragments (<4 words) are dropped immediately — they are almost
        never complete instructions.
        """
        raw = re.split(r'[\r\n]+|(?<=[.!?])\s+', text)
        cleaned = []
        for line in raw:
            line = line.strip(" -•*\t")
            # Strip leading numbered/lettered list prefixes: "1.", "2)", "(a)", "a."
            line = re.sub(r'^\s*(\d+[\.\)]\s*|[a-zA-Z][\.\)]\s*|\([a-zA-Z0-9]\)\s*)', '', line).strip()
            if len(line.split()) >= 4:   # drop fragments under 4 words
                cleaned.append(line)
        return cleaned

    @staticmethod
    def _is_noise(line: str) -> bool:
        """Return True if the line matches a known noise/filler pattern."""
        for pattern in _NOISE_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _is_option_def(line: str) -> bool:
        return bool(re.match(r"^/[A-Z]", line.strip(), re.IGNORECASE))

    @staticmethod
    def _extract_option_name(line: str) -> str:
        m = re.match(r"^(\/[A-Z]+(?:\[[^\]]+\])?)", line.strip(), re.IGNORECASE)
        return m.group(1).upper() if m else ""

    @staticmethod
    def _extract_option_description(line: str) -> str:
        m = re.match(r"^\/[A-Z]+(?:\[[^\]]+\])?\s+(.*)", line.strip(), re.IGNORECASE)
        return m.group(1).strip() if m else line.strip()

    @staticmethod
    def _is_actionable(line: str) -> bool:
        """
        Three-tier filter
          1 Drop known noise patterns immediately
          2 Pass lines that match a strong actionable pattern
          3 Pass lines that match 2+ weak patterns (reduces filler false positives)
           4 Drop everything else
        """
        # Tier 1  noise gate
        if RubricHelperApp._is_noise(line):
            return False

        # Option definitions always pass
        if RubricHelperApp._is_option_def(line):
            return True

        lowered = line.lower()

        # Tier 2  strong patterns one match is enough
        for pattern in _STRONG_PATTERNS:
            if re.search(pattern, lowered):
                return True

        # Tier 3  weak patterns need at least 2 matches
        weak_hits = sum(1 for p in _WEAK_PATTERNS if re.search(p, lowered))
        if weak_hits >= 2:
            return True

        return False

    @staticmethod
    def _extract_languages(line: str) -> List[str]:
        found = []
        for lang in ["C++", "Java", "Python", "C"]:
            if re.search(rf"\b{re.escape(lang)}\b", line, re.IGNORECASE):
                found.append(lang)
        return found

    @staticmethod
    def _is_language_choice_line(line: str) -> bool:
        """
        Returns True if the line is primarily about which language to use,
        not just mentioning a language name in passing.

        "You may use Python, Java, or C++" -> True  (language choice)
        "Write a Python program that..."   -> False (main task that mentions Python)
        """
        lowered = line.lower()
        language_choice_signals = [
            r"\buse\s+(any|one\s+of|the\s+following)\b",  # use any of, use one of
            r"\b(may|can|must)\s+use\b",                     # you may use, must use
            r"\ballowed\s+(language|to\s+use)\b",           # allowed language
            r"\b(programming\s+)?language(s)?\s*:",          # languages: Python...
            r"\bchoose\s+(any|one|from)\b",                  # hoose one of
            r"\bin\s+(python|java|c\+\+|c)\b",             # implement in Python
        ]
        # Also check: if line has 2+ languages listed together it's a choice line
        lang_count = sum(1 for lang in ["python", "java", "c++", " c "]
                         if lang in lowered)
        if lang_count >= 2:
            return True
        return any(re.search(p, lowered) for p in language_choice_signals)

    @staticmethod
    def _force_bucket(line: str) -> str:
        """
        Rule-based pre-classifier for lines that are unambiguously one category.
        Returns a bucket key if confident, or empty string to fall through to NLI.

        IMPORTANT: Exception/negative signals are checked FIRST.
        A line like "you do not need to implement X" contains "implement" but is
        an exception, not a main task. Checking exceptions first prevents this.
        """
        lowered = line.lower()

        # Tier 0 Exception signals  checked before everything else 
        # Any line with a negation + action word is an exception, not a task.
        exception_signals = [
            r"\bdo\s+not\s+(need|have)\s+to\b",   # do not need to, do not have to
            r"\bdoes\s+not\s+need\s+to\b",
            r"\bnot\s+required\s+to\b",
            r"\bnot\s+necessary\s+to\b",
            r"\bneed\s+not\b",
            r"\byou\s+do\s+not\b",                  # you do not need/have to
            r"\byou\s+don't\b",
            r"\bno\s+need\s+to\b",
            r"\bexclude\b", r"\bskip\b",
            r"^\s*note\s*:",                           # "Note: ..."
            r"^\s*exception\s*:",
        ]
        for p in exception_signals:
            if re.search(p, lowered):
                return "exception_note"

        #  Tier 1Submission signals 
        submission_signals = [
            r"\bsubmit\b", r"\bsubmission\b", r"\bturn\s+in\b",
            r"\bupload\b", r"\bcanvas\b", r"\bgradescope\b",
            r"\bblackboard\b",
            r"\bdue\s+(date|by|on|friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b",
            r"\bdeadline\b", r"\blate\s+(submission|penalty|work)\b",
            r"\bby\s+(friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b",
            r"\b\d{1,2}:\d{2}\s*(am|pm)\b",
            r"\b11:59\b",
        ]
        for p in submission_signals:
            if re.search(p, lowered):
                return "submission_requirement"

        # Tier 2  Main task signals
        # Only if the line is clearly a positive instruction, not a negation.
        main_task_signals = [
            r"\bwrite\s+a\b", r"\bwrite\s+an\b",
            r"\bcreate\s+a\b", r"\bcreate\s+an\b",
            r"\bbuild\s+a\b", r"\bbuild\s+an\b",
            r"\bimplement\s+a\b", r"\bimplement\s+an\b",
            r"\bdevelop\s+a\b", r"\bdevelop\s+an\b",
            r"\bprogram\s+that\b", r"\bprogram\s+which\b",
            r"\bapplication\s+that\b",
            r"\bsimulate\s+a\b", r"\bsimulate\s+an\b",
        ]
        for p in main_task_signals:
            if re.search(p, lowered):
                return "main_task"

        # Tier 3  Output requirement signals
        output_signals = [
            r"\boutput\s+(must|should|needs?\s+to|will)\b",
            r"\bthe\s+output\s+(must|should|will|needs?\s+to)\b",
            r"\bprint\s+(the|a|each|all|every|results?|values?|numbers?|list|names?)\b",
            r"\bdisplay\s+(the|a|each|all|results?|output|values?)\b",
            r"\bformat\s+of\s+the\s+(output|result)\b",
            r"\bexact\s+(output|format|same\s+format)\b",
            r"\bmatch\s+the\s+expected\s+(output|format)\b",
            r"\bone\s+(result|item|entry|number|value|name)\s+per\s+line\b",
            r"\bone\s+per\s+line\b",
            r"\bsorted\s+(in|by)\s+(ascending|descending)\b",
            r"\bin\s+ascending\s+order\b", r"\bin\s+descending\s+order\b",
            r"\balphabetical(ly)?\s+order\b",
            r"\bno\s+trailing\s+(spaces?|newlines?)\b",
            r"\bfollowed\s+by\s+a\s+newline\b",
            r"\bseparated\s+by\s+(a\s+)?(space|comma|newline|tab)\b",
        ]
        for p in output_signals:
            if re.search(p, lowered):
                return "output_requirement"

        # Tier 4  Supported option/feature signals
        option_signals = [
            r"\bcommand[\s-]line\s+(argument|flag|option|parameter)s?\b",
            r"\baccept\s+.{0,30}(flag|option|argument|switch)\b",
            r"\bsupport\s+(the\s+)?-[a-z]\b",
            r"\b-[a-z]\s+(flag|option|switch)\b",
            r"\bargc\b", r"\bargv\b", r"\bgetopt\b",
            r"\bflag\s+to\b",
            r"\bverbose\s+(mode|flag|output)\b",
            r"\b(enable|disable)\s+.{0,20}(flag|option|mode)\b",
        ]
        for p in option_signals:
            if re.search(p, lowered):
                return "supported_option"

        return ""

    # Main pipeline

    def _analyze(self, text: str) -> Tuple[Dict, Set[str]]:
        lines = self._split_lines(text)
        actionable = [l for l in lines if self._is_actionable(l)]

        buckets: Dict[str, List[dict]] = {k: [] for k in _BUCKET_KEYS}
        buckets["unclassified"] = []

        options: List[dict] = []
        languages: Set[str] = set()

        total = len(actionable)
        for i, line in enumerate(actionable):
            self._set_status(f"Classifying line {i + 1} of {total}…")

            # Option definitions (/V, /C …) — rule-based, no NLI needed
            if self._is_option_def(line):
                name = self._extract_option_name(line)
                desc = self._extract_option_description(line)
                if name:
                    options.append({"name": name, "description": desc})
                continue

            # Language detection — only force to allowed_language if the line
            # is primarily ABOUT language choice (e.g. use Python, Java, or C++").
            # If the line just mentions a language name in passing (e.g. "Write a
            # Python program that..."), collect the language but let the line fall
            # through to _force_bucket and NLI for proper classification.
            langs = self._extract_languages(line)
            if langs:
                for lang in langs:
                    languages.add(lang)
                if RubricHelperApp._is_language_choice_line(line):
                    _, score, all_scores = self._classify(line)
                    buckets["allowed_language"].append({
                        "text": line,
                        "confidence": score,
                        "all_scores": all_scores,
                    })
                    continue
                # else: language mentioned in passing — fall through to normal classification

            # Rule-based override for high-confidence cases
            # These patterns are so unambiguous we skip NLI entirely
            forced = RubricHelperApp._force_bucket(line)
            if forced:
                buckets[forced].append({
                    "text": line,
                    "confidence": 1.0,
                    "all_scores": {},
                    "is_rule_based": True,
                })
                continue

            # NLI zero-shot classification
            bucket_key, score, all_scores = self._classify(line)
            buckets[bucket_key].append({
                "text": line,
                "confidence": score,
                "all_scores": all_scores,
            })

        # Consolidate all detected options into one formatted entry
        if options:
            parts = ", ".join(f"{o['name']} — {o['description']}" for o in options)
            buckets["supported_option"].append({
                "text": f"Support these command options: {parts}",
                "confidence": 1.0,
                "all_scores": {},
                "is_rule_based": True,
            })

        return buckets, languages

    #  Rendering 

    def _render(self, buckets: Dict, languages: Set[str]):
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)

        any_output = False
        order = [
            "main_task", "output_requirement", "supported_option",
            "allowed_language", "exception_note", "submission_requirement",
        ]

        for key in order:
            items = list(buckets.get(key, []))

            # Merge language detections into one clean summary line
            if key == "allowed_language" and languages:
                lang_line = "Allowed languages: " + ", ".join(sorted(languages))
                conf = max((i["confidence"] for i in items), default=1.0)
                items = [{"text": lang_line, "confidence": conf, "all_scores": {}}]

            if not items:
                continue

            any_output = True
            meta = BUCKETS[key]
            self.results_text.insert(tk.END, f"\n{meta['display']}\n", f"header_{key}")
            self.results_text.insert(tk.END, "─" * 52 + "\n", "rule")

            for item in items:
                is_rule = item.get("is_rule_based", False)
                pct = int(item["confidence"] * 100)
                conf_label = "[rule]" if is_rule else f"[{pct:3d}%]"
                bar = "" if is_rule else "  " + self._conf_bar(item["confidence"])

                self.results_text.insert(tk.END, "  • ", f"item_{key}")
                self.results_text.insert(tk.END, item["text"] + "\n", f"item_{key}")
                self.results_text.insert(
                    tk.END,
                    f"         {conf_label}{bar}\n",
                    f"conf_{key}",
                )

                # Show runner-up scores so users can see why a line landed here
                if item["all_scores"] and not is_rule:
                    sorted_scores = sorted(
                        item["all_scores"].items(), key=lambda x: x[1], reverse=True
                    )
                    for rank, (bk, sc) in enumerate(sorted_scores[:3]):
                        marker = "▶" if rank == 0 else " "
                        label = BUCKETS[bk]["display"] if bk in BUCKETS else bk
                        self.results_text.insert(
                            tk.END,
                            f"           {marker} {label}: {int(sc * 100)}%\n",
                            f"conf_{key}",
                        )

        # Unclassified
        unclassified = buckets.get("unclassified", [])
        if unclassified:
            any_output = True
            self.results_text.insert(tk.END, "\nOther / Unclassified\n", "unclassified_header")
            self.results_text.insert(tk.END, "─" * 52 + "\n", "rule")
            for item in unclassified:
                self.results_text.insert(tk.END, f"  • {item['text']}\n", "unclassified_item")
                pct = int(item["confidence"] * 100)
                self.results_text.insert(
                    tk.END,
                    f"         [best score was only {pct}% — below threshold]\n",
                    "unclassified_item",
                )

        if not any_output:
            self.results_text.insert(tk.END, "No clear requirements detected.", "loading")

        self.results_text.configure(state="disabled")

    @staticmethod
    def _conf_bar(score: float, width: int = 18) -> str:
        filled = round(score * width)
        return "█" * filled + "░" * (width - filled) + f"  {int(score * 100)}%"

    #  Event handlers 

    def _on_analyze(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text or self.input_placeholder_active:
            messagebox.showwarning("Empty Input", "Please paste a rubric or upload a PDF.")
            return
        if not self.model_ready:
            messagebox.showinfo("Not Ready", "Model is still loading.")
            return

        self.analyze_btn.state(["disabled"])
        self.progress.start(10)

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Running NLI classification…\n", "loading")
        self.results_text.configure(state="disabled")
        self.root.update()

        def run():
            try:
                buckets, languages = self._analyze(text)
                self.root.after(0, lambda: self._render(buckets, languages))
                self.root.after(0, lambda: self._set_status("✓ Done"))
            except Exception as exc:
                err_msg = str(exc)
                self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
            finally:
                self.root.after(0, self.progress.stop)
                self.root.after(0, lambda: self.analyze_btn.state(["!disabled"]))

        threading.Thread(target=run, daemon=True).start()

    def _clear(self):
        self.input_text.delete("1.0", tk.END)
        self.input_text.configure(fg="#0f172a")
        self.input_placeholder_active = False
        self._set_placeholder()
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.configure(state="disabled")

    def _get_results_content(self) -> str:
        return self.results_text.get("1.0", tk.END).strip()

    def _copy_results(self):
        text = self._get_results_content()
        if not text:
            messagebox.showinfo("No Results", "There is no extracted text to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status("✓ Extracted requirements copied to clipboard")

    def _save_results(self):
        text = self._get_results_content()
        if not text:
            messagebox.showinfo("No Results", "There is no extracted text to save yet.")
            return

        path = filedialog.asksaveasfilename(
            title="Save extracted requirements",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            self._set_status(f"✓ Saved extracted requirements to {path}")
        except Exception as exc:
            messagebox.showerror("Save Error", f"Could not save file:\n{exc}")


def main():
    root = tk.Tk()
    RubricHelperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
