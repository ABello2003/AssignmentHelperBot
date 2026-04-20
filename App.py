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
        "label": "write, build, create, develop, or implement a program, application, or system",
        "display": "Main Task",
        "color": "#1a56db",
    },
    "output_requirement": {
        "label": "a requirement about how the program output should look or behave",
        "display": "Output Requirement",
        "color": "#7e3af2",
    },
    "supported_option": {
        "label": "a command-line option, flag, or feature the program must support",
        "display": "Supported Options / Features",
        "color": "#0694a2",
    },
    "allowed_language": {
        "label": "which programming languages are allowed or permitted to use",
        "display": "Allowed Languages",
        "color": "#057a55",
    },
    "exception_note": {
        "label": "something that is not required, optional, or explicitly excluded",
        "display": "Exceptions / Notes",
        "color": "#d97706",
    },
    "submission_requirement": {
        "label": "submission deadline, due date, when to turn in, upload to Canvas, late penalty, or submission instructions",
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

        self._build_gui()
        threading.Thread(target=self._load_model, daemon=True).start()

    #  GUI 

    def _build_gui(self):
        style = ttk.Style()
        style.configure("TFrame", background="#f8f8f8")
        style.configure("TLabel", background="#f8f8f8")
        style.configure("Bold.TLabel", background="#f8f8f8", font=("Arial", 11, "bold"))
        style.configure("Title.TLabel", background="#f8f8f8", font=("Arial", 17, "bold"))
        style.configure("Sub.TLabel", background="#f8f8f8", font=("Arial", 10), foreground="#555")

        main = ttk.Frame(self.root, padding=14)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Rubric Helper", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            main,
            text=f"Zero-shot NLI classification via {MODEL_NAME}",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 10))

        #  Input 
        ttk.Label(main, text="Assignment Prompt / Rubric:", style="Bold.TLabel").pack(anchor="w")
        self.input_text = scrolledtext.ScrolledText(
            main, wrap=tk.WORD, height=10, font=("Arial", 11), relief="solid", borderwidth=1
        )
        self.input_text.pack(fill="x", pady=(4, 8))

        #  Buttons 
        btn_row = ttk.Frame(main)
        btn_row.pack(fill="x", pady=(0, 8))

        self.analyze_btn = ttk.Button(btn_row, text="Analyze", command=self._on_analyze)
        self.analyze_btn.pack(side="left")
        self.analyze_btn.state(["disabled"])

        # PDF upload button
        self.pdf_btn = ttk.Button(btn_row, text="Upload PDF", command=self._upload_pdf)
        self.pdf_btn.pack(side="left", padx=8)

        ttk.Button(btn_row, text="Clear", command=self._clear).pack(side="left")

        self.status_var = tk.StringVar(value="Loading NLI model…")
        ttk.Label(btn_row, textvariable=self.status_var, style="Sub.TLabel").pack(
            side="left", padx=12
        )

        # Legend 
        legend = ttk.Frame(main)
        legend.pack(anchor="w", pady=(0, 6))
        for key, meta in BUCKETS.items():
            tk.Label(legend, text="●", fg=meta["color"], bg="#f8f8f8", font=("Arial", 12)).pack(side="left")
            tk.Label(legend, text=meta["display"] + "  ", bg="#f8f8f8", font=("Arial", 9)).pack(side="left")

        #  Results 
        ttk.Label(main, text="Extracted Requirements:", style="Bold.TLabel").pack(anchor="w")
        self.results_text = scrolledtext.ScrolledText(
            main,
            wrap=tk.WORD,
            height=18,
            font=("Courier New", 11),
            relief="solid",
            borderwidth=1,
            state="disabled",
        )
        self.results_text.pack(fill="both", expand=True, pady=(4, 0))

        for key, meta in BUCKETS.items():
            self.results_text.tag_config(f"header_{key}", foreground=meta["color"],
                                         font=("Arial", 11, "bold"))
            self.results_text.tag_config(f"item_{key}", foreground="#1a1a1a",
                                         font=("Courier New", 11))
            self.results_text.tag_config(f"conf_{key}", foreground=meta["color"],
                                         font=("Courier New", 9))

        self.results_text.tag_config("unclassified_header", foreground="#888",
                                     font=("Arial", 11, "bold"))
        self.results_text.tag_config("unclassified_item", foreground="#888",
                                     font=("Courier New", 11))
        self.results_text.tag_config("loading", foreground="#555", font=("Arial", 11, "italic"))
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
            self.input_text.delete("1.0", tk.END)
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
        self._set_status(f"✓ {MODEL_NAME} ready")

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self.status_var.set(msg))

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

        # 2nd tier Main task signals 
        # Only  if the line is clearly a positive instruction, not a negation.
        main_task_signals = [
            r"\bwrite\s+a\b", r"\bwrite\s+an\b",
            r"\bcreate\s+a\b", r"\bcreate\s+an\b",
            r"\bbuild\s+a\b", r"\bbuild\s+an\b",
            r"\bimplement\s+a\b", r"\bimplement\s+an\b",
            r"\bdevelop\s+a\b", r"\bdevelop\s+an\b",
            r"\bprogram\s+that\b", r"\bprogram\s+which\b",
            r"\bapplication\s+that\b",
        ]
        for p in main_task_signals:
            if re.search(p, lowered):
                return "main_task"

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
        if not text:
            messagebox.showwarning("Empty Input", "Please paste a rubric or upload a PDF.")
            return
        if not self.model_ready:
            messagebox.showinfo("Not Ready", "Model is still loading.")
            return

        self.analyze_btn.state(["disabled"])

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
                self.root.after(0, lambda: self.analyze_btn.state(["!disabled"]))

        threading.Thread(target=run, daemon=True).start()

    def _clear(self):
        self.input_text.delete("1.0", tk.END)
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.configure(state="disabled")


def main():
    root = tk.Tk()
    RubricHelperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
