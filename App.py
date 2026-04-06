import re
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import List, Dict, Set, Tuple

from transformers import pipeline



# cross-encoder/nli-deberta-v3-small is a pretrained NLI (Natural Language Inference) classifier from Hugging Face. , 
# model reads both  input line n the candidate label together and outputs an entailment probability reasoning about how well the line mathes its labels

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Minimum entailment probability to accept a classification.

CONFIDENCE_THRESHOLD = 0.40


#Buckets are pre-defined categories we want to sort rubric lines into. Each bucket has a human-friendly label, a display name for the UI, and a color for rendering results.
# label strings get passed directly to the NLI model as a candidate
#  Phrasing them as hypothesis statements ("This sentence is about X") 
BUCKETS = {
    "main_task": {
        "label": "the main programming task or deliverable to build or implement",
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
    "exception_note" : {
        "label": "something that is not required, optional, or explicitly excluded",
        "display": "Exceptions / Notes",
        "color": "#d97706",
    },
    "submission_requirement": {
        "label": "how or when to submit the assignment, deadline, or upload instructions",
        "display": "Submission",
        "color": "#e02424",
    },
}


_BUCKET_KEYS =  list(BUCKETS.keys())
_CANDIDATE_LABELS = [BUCKETS[k]["label"] for k in _BUCKET_KEYS]


class RubricHelperApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Rubric Helper — NLI Classifier")
        self.root.geometry("1100x780")
        self.root.configure(bg="#f8f8f8")

        self.classifier = None   
        self.model_ready = False

        self._build_gui()
        threading.Thread(target=self._load_model, daemon=True).start()

    # GUI

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

        # Input 
        ttk.Label(main, text="Assignment Prompt / Rubric:", style="Bold.TLabel").pack(anchor="w")
        self.input_text = scrolledtext.ScrolledText(
            main, wrap=tk.WORD, height=11, font=("Arial", 11), relief="solid", borderwidth=1
        )
        self.input_text.pack(fill="x", pady=(4, 8))
        

        # Buttons  for analyzing and clearing input, plus a status label to show model loading progress and other messages
        btn_row = ttk.Frame(main)
        btn_row.pack(fill="x", pady=(0, 8))

        self.analyze_btn = ttk.Button(btn_row, text="Analyze", command=self._on_analyze)
        self.analyze_btn.pack(side="left")
        self.analyze_btn.state(["disabled"])

        ttk.Button(btn_row, text="Clear", command=self._clear).pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="Loading NLI model…")
        ttk.Label(btn_row, textvariable=self.status_var, style="Sub.TLabel").pack(
            side="left", padx=8
        )

        # Legend for bucket colors
        legend = ttk.Frame(main)
        legend.pack(anchor="w", pady=(0, 6))
        for key, meta in BUCKETS.items():
            tk.Label(legend, text="●", fg=meta["color"], bg="#f8f8f8", font=("Arial", 12)).pack(side="left")
            tk.Label(legend, text=meta["display"] + "  ", bg="#f8f8f8", font=("Arial", 9)).pack(side="left")

        # Results for eah bucket, plus an Other / Unclassified section at the end for lines that don't meet the confidence threshold
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

    #  Model loading 

    def _load_model(self):
        try:
            self._set_status("Downloading model weights… (first run only, ~180MB)")
            # pipelinezero-shot-classification wraps the NLI model so we can pass any text + candidate labels and get back entailment probabilities.
            self.classifier = pipeline(
                "zero-shot-classification",
                model=MODEL_NAME,
                device=-1,   # CPU; change to 0 for GPU
            )
            self.model_ready = True
            self.root.after(0, self._on_model_ready)
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Model Error", str(exc)))

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
            premise   = the rubric line
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

        # Map label strings back to bucket keys
        all_scores: Dict[str, float] = {}
        for label, score in zip(result["labels"], result["scores"]):
            idx = _CANDIDATE_LABELS.index(label)
            all_scores[_BUCKET_KEYS[idx]] = score

        best_key = max(all_scores, key=all_scores.__getitem__)
        best_score = all_scores[best_key]

        if best_score < CONFIDENCE_THRESHOLD:
            return "unclassified", best_score, all_scores

        return best_key, best_score, all_scores

    # Text processing 

    @staticmethod
    def _split_lines(text: str) -> List[str]:
        raw = re.split(r'[\r\n]+|(?<=[.!?])\s+', text)
        return [l.strip(" -•*\t") for l in raw if l.strip(" -•*\t")]

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
        """Pre-filter: drop lines that are clearly background/descriptive filler."""
        lowered = line.lower()
        patterns = [
            r"\bwrite\b", r"\bcreate\b", r"\bbuild\b", r"\bimplement\b",
            r"\bdevelop\b", r"\bsimulate\b", r"\bhandle\b", r"\buse\b",
            r"\binclude\b", r"\bsubmit\b", r"\bmust\b", r"\bshould\b",
            r"\bdo not have to\b", r"\bdoes not need\b", r"\bneed not\b",
            r"\bnote\b", r"\bsupport\b", r"\ballow\b", r"\brequire\b",
            r"\bproduce\b", r"\bdisplay\b", r"\boutput\b", r"\bupload\b",
            r"\bturn in\b", r"\bdeadline\b",
        ]
        if RubricHelperApp._is_option_def(line):
            return True
        return any(re.search(p, lowered) for p in patterns)

    @staticmethod
    def _extract_languages(line: str) -> List[str]:
        found = []
        for lang in ["C++", "Java", "Python", "C"]:
            if re.search(rf"\b{re.escape(lang)}\b", line, re.IGNORECASE):
                found.append(lang)
        return found

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

           
            if self._is_option_def(line):
                name = self._extract_option_name(line)
                desc = self._extract_option_description(line)
                if name:
                    options.append({"name": name, "description": desc})
                continue

            #  Language detection  rule-based override 
            langs = self._extract_languages(line)
            if langs:
                for lang in langs:
                    languages.add(lang)
                # Run NLI anyway so we can show a real confidence score
                _, score, all_scores = self._classify(line)
                buckets["allowed_language"].append({
                    "text": line,
                    "confidence": score,
                    "all_scores": all_scores,
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
                    tk.END, f"         [best score was only {pct}% — below threshold]\n",
                    "unclassified_item"
                )

        if not any_output:
            self.results_text.insert(tk.END, "No clear requirements detected.", "loading")

        self.results_text.configure(state="disabled")

    @staticmethod
    def _conf_bar(score: float, width: int = 18) -> str:
        filled = round(score * width)
        return "█" * filled + "░" * (width - filled) + f"  {int(score * 100)}%"

    # Event handlers 

    def _on_analyze(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Empty Input", "Please paste a rubric or prompt.")
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
                self.root.after(0, lambda: messagebox.showerror("Error", str(exc)))
            finally:
                self.root.after(0, lambda: self.analyze_btn.state(["!disabled"]))

        threading.Thread(target=run, daemon=True).start()

    def _clear(self):
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.configure(state="disabled")





def main():
    root = tk.Tk()
    RubricHelperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
