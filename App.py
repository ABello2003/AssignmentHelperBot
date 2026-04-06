import re
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import List, Dict, Set

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# BERT model used for semantic grouping
MODEL_NAME = "bert-base-uncased"


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token embeddings into one sentence embedding."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class RubricHelperApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Rubric Helper")
        self.root.geometry("1050x720")

        self.tokenizer = None
        self.model = None
        self.bucket_embeddings = {}

        # Semantic buckets used to organize requirement lines
        self.bucket_descriptions = {
            "main_task": "write build create implement develop simulate program application command system",
            "output_requirement": "output should look the same exact format same result behave exactly like original",
            "supported_option": "support handle command option flag switch parameter argument",
            "allowed_language": "allowed language use c c++ java python programming language",
            "exception_note": "do not need not required exception note optional skip exclude",
            "submission_requirement": "submit upload turn in final file pdf report presentation",
        }

        self.build_gui()
        self.load_model()

    def build_gui(self):
        """Create the Tkinter interface."""
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        title_label = ttk.Label(
            main_frame,
            text="Rubric Helper",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=(0, 10))

        description_label = ttk.Label(
            main_frame,
            text="Paste a rubric or assignment prompt below, then click Analyze.",
            font=("Arial", 11)
        )
        description_label.pack(pady=(0, 10))

        input_label = ttk.Label(
            main_frame,
            text="Rubric / Assignment Prompt:",
            font=("Arial", 11, "bold")
        )
        input_label.pack(anchor="w")

        self.input_text = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            height=14,
            font=("Arial", 11)
        )
        self.input_text.pack(fill="x", pady=(5, 10))

        sample_text = (
            "Write a program that simulates the Find command.\n"
            "Handle all of the possible options available in the Find command.\n"
            "It must work exactly like the Find command.\n"
            "The output should look the same as well.\n"
            "Note: You do not have to handle the [/OFF[LINE]] option.\n"
            "Use any of these 4 languages: C, C++, Java, Python.\n"
            "/V Displays all lines NOT containing the specified string.\n"
            "/C Displays only the count of lines containing the string.\n"
            "/N Displays line numbers with the displayed lines.\n"
            "/I Ignores the case of characters when searching for the string."
        )
        self.input_text.insert("1.0", sample_text)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(0, 10))

        analyze_button = ttk.Button(
            button_frame,
            text="Analyze",
            command=self.analyze_prompt
        )
        analyze_button.pack(side="left")

        clear_button = ttk.Button(
            button_frame,
            text="Clear Results",
            command=self.clear_results
        )
        clear_button.pack(side="left", padx=(10, 0))

        results_label = ttk.Label(
            main_frame,
            text="Simplified Requirements:",
            font=("Arial", 11, "bold")
        )
        results_label.pack(anchor="w")

        self.results_text = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            height=18,
            font=("Arial", 11)
        )
        self.results_text.pack(fill="both", expand=True, pady=(5, 0))

    def load_model(self):
        """
        Load BERT and precompute embeddings for semantic buckets.
        BERT is used here to group candidate lines into requirement types.
        """
        try:
            self.results_text.insert(tk.END, "Loading BERT model. Please wait...\n")
            self.root.update()

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModel.from_pretrained(MODEL_NAME)
            self.model.eval()

            for bucket_name, description in self.bucket_descriptions.items():
                self.bucket_embeddings[bucket_name] = self.embed_text(description)

            self.results_text.insert(tk.END, "Model loaded successfully.\n\n")

        except Exception as e:
            messagebox.showerror("Error", f"Could not load model:\n{e}")

    def embed_text(self, text: str) -> torch.Tensor:
        """Convert text into a normalized BERT embedding."""
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            output = self.model(**encoded)

        pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
        normalized = F.normalize(pooled, p=2, dim=1)
        return normalized.squeeze(0)

    def cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Compute similarity between two normalized embeddings."""
        return torch.dot(vec1, vec2).item()

    def split_lines(self, text: str) -> List[str]:
        """
        Split the prompt into manageable lines.
        We keep line breaks because rubrics are often line-oriented.
        """
        raw_lines = re.split(r'[\r\n]+|(?<=[.!?])\s+', text)
        cleaned = []

        for line in raw_lines:
            line = line.strip(" -•\t")
            if line:
                cleaned.append(line)

        return cleaned

    def is_option_definition(self, line: str) -> bool:
        """Check if a line defines an option like /V or /C."""
        return bool(re.match(r"^/\w+", line.strip()))

    def extract_option_name(self, line: str) -> str:
        """Extract the option name from a line like '/V Displays ...'."""
        match = re.match(r"^(\/[A-Z]+(?:\[[A-Z]+\])?)", line.strip(), flags=re.IGNORECASE)
        return match.group(1) if match else ""

    def is_actionable_line(self, line: str) -> bool:
        """
        Keep lines that look like actual requirements.
        Ignore background lines that only explain or describe.
        """
        lowered = line.lower()

        action_patterns = [
            r"\bwrite\b",
            r"\bcreate\b",
            r"\bbuild\b",
            r"\bimplement\b",
            r"\bdevelop\b",
            r"\bsimulate\b",
            r"\bhandle\b",
            r"\buse\b",
            r"\binclude\b",
            r"\bsubmit\b",
            r"\bmust\b",
            r"\bshould\b",
            r"\bdo not have to\b",
            r"\bdoes not need\b",
            r"\bneed not\b",
            r"\bnote\b",
        ]

        if self.is_option_definition(line):
            return True

        return any(re.search(pattern, lowered) for pattern in action_patterns)

    def assign_bucket(self, line: str) -> str:
        """
        Use BERT embeddings to assign an actionable line to a semantic bucket.
        """
        line_vec = self.embed_text(line)

        best_bucket = None
        best_score = -1.0

        for bucket_name, bucket_vec in self.bucket_embeddings.items():
            score = self.cosine_similarity(line_vec, bucket_vec)
            if score > best_score:
                best_score = score
                best_bucket = bucket_name

        return best_bucket

    def extract_languages(self, line: str) -> List[str]:
        """Pull programming languages from a sentence."""
        found = []
        for lang in ["C++", "Java", "Python", "C"]:
            if re.search(rf"\b{re.escape(lang)}\b", line, flags=re.IGNORECASE):
                found.append(lang)
        return found

    def clean_task_text(self, line: str) -> str:
        """Make a line more readable in the final summary."""
        line = re.sub(r"\s+", " ", line).strip()
        return line.rstrip(".")

    def summarize_requirements(self, text: str) -> Dict[str, List[str]]:
        """
        Main extraction pipeline:
        1. Keep only actionable lines
        2. Use BERT to group them
        3. Consolidate into a short checklist
        """
        lines = self.split_lines(text)
        actionable_lines = [line for line in lines if self.is_actionable_line(line)]

        main_tasks: List[str] = []
        requirements: List[str] = []
        exceptions: List[str] = []
        submissions: List[str] = []
        languages: Set[str] = set()
        options: Set[str] = set()

        for line in actionable_lines:
            # If the line is an option definition like "/V Displays ..."
            # keep only the option name for concise output.
            if self.is_option_definition(line):
                option_name = self.extract_option_name(line)
                if option_name:
                    options.add(option_name.upper())
                continue

            bucket = self.assign_bucket(line)
            cleaned_line = self.clean_task_text(line)

            if bucket == "allowed_language":
                for lang in self.extract_languages(line):
                    languages.add(lang)

            elif bucket == "exception_note":
                exceptions.append(cleaned_line)

            elif bucket == "submission_requirement":
                submissions.append(cleaned_line)

            elif bucket == "output_requirement":
                requirements.append(cleaned_line)

            elif bucket == "supported_option":
                requirements.append(cleaned_line)

            elif bucket == "main_task":
                main_tasks.append(cleaned_line)

            else:
                requirements.append(cleaned_line)

        # Consolidate option handling into one clean bullet
        if options:
            ordered_options = sorted(options)
            requirements.append(
                "Support these command options: " + ", ".join(ordered_options)
            )

        # Deduplicate while preserving order
        def unique_keep_order(items: List[str]) -> List[str]:
            seen = set()
            output = []
            for item in items:
                key = item.lower()
                if key not in seen:
                    seen.add(key)
                    output.append(item)
            return output

        return {
            "Main Task": unique_keep_order(main_tasks),
            "Requirements": unique_keep_order(requirements),
            "Allowed Languages": sorted(languages),
            "Submission": unique_keep_order(submissions),
            "Exceptions / Notes": unique_keep_order(exceptions),
        }

    def format_summary(self, summary: Dict[str, List[str]]) -> str:
        """Convert the extracted summary into readable text."""
        sections = []

        for title, items in summary.items():
            if not items:
                continue

            sections.append(title)
            sections.append("-" * len(title))

            for item in items:
                sections.append(f"• {item}")

            sections.append("")

        if not sections:
            return "No clear requirements were found."

        return "\n".join(sections)

    def analyze_prompt(self):
        """Analyze the rubric and display the simplified checklist."""
        user_text = self.input_text.get("1.0", tk.END).strip()

        if not user_text:
            messagebox.showwarning("Missing Input", "Please paste a rubric or assignment prompt first.")
            return

        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Analyzing rubric...\n\n")
        self.root.update()

        try:
            summary = self.summarize_requirements(user_text)
            formatted = self.format_summary(summary)

            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, formatted)

        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong:\n{e}")

    def clear_results(self):
        """Clear the results area."""
        self.results_text.delete("1.0", tk.END)


def main():
    root = tk.Tk()
    app = RubricHelperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
