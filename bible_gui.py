"""A lightweight Tkinter GUI for the Bible RAG pipeline."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Sequence

from bible_rag import BibleRAGPipeline, RetrievedPassage


class BibleAssistantGUI(tk.Tk):
    """Interactive window for querying the Bible RAG pipeline."""

    def __init__(
        self,
        pipeline: BibleRAGPipeline,
        *,
        default_top_k: int = 5,
        reasoning_effort: str = "none",
        text_verbosity: str = "medium",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        show_passages_default: bool = True,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.title("KJV Bible Assistant")
        self.minsize(900, 650)
        self._build_layout(default_top_k, show_passages_default)

    def _build_layout(self, default_top_k: int, show_passages_default: bool) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        container = ttk.Frame(self, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)
        container.rowconfigure(4, weight=1)
        container.rowconfigure(6, weight=1)

        ttk.Label(container, text="Question").grid(row=0, column=0, sticky="w")
        self.question_text = tk.Text(container, height=5, wrap="word")
        self.question_text.grid(row=1, column=0, sticky="nsew", pady=(4, 0))

        controls = ttk.Frame(container)
        controls.grid(row=2, column=0, sticky="ew", pady=(12, 8))
        controls.columnconfigure(3, weight=1)

        ttk.Label(controls, text="Top K:").grid(row=0, column=0, sticky="w")
        self.top_k_var = tk.IntVar(value=max(1, default_top_k))
        top_k_spin = ttk.Spinbox(
            controls,
            from_=1,
            to=20,
            width=6,
            textvariable=self.top_k_var,
            increment=1,
        )
        top_k_spin.grid(row=0, column=1, sticky="w", padx=(6, 24))

        self.show_passages_var = tk.BooleanVar(value=show_passages_default)
        ttk.Checkbutton(
            controls,
            text="Show passages",
            variable=self.show_passages_var,
        ).grid(row=0, column=2, sticky="w")

        button_row = ttk.Frame(controls)
        button_row.grid(row=0, column=3, sticky="e")
        self.ask_button = ttk.Button(button_row, text="Ask", command=self.on_ask)
        self.ask_button.grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_row, text="Clear", command=self.clear_fields).grid(row=0, column=1)

        ttk.Label(container, text="Answer").grid(row=3, column=0, sticky="w")
        self.answer_text = tk.Text(container, height=10, wrap="word", state="disabled")
        self.answer_text.grid(row=4, column=0, sticky="nsew", pady=(4, 0))

        ttk.Label(container, text="Retrieved Passages").grid(row=5, column=0, sticky="w", pady=(12, 0))
        self.passages_text = tk.Text(container, height=10, wrap="word", state="disabled")
        self.passages_text.grid(row=6, column=0, sticky="nsew", pady=(4, 0))

        self.status_var = tk.StringVar(value="Enter a question and press Ask.")
        ttk.Label(container, textvariable=self.status_var, foreground="gray").grid(
            row=7, column=0, sticky="w", pady=(12, 0)
        )

    def on_ask(self) -> None:
        question = self.question_text.get("1.0", tk.END).strip()
        if not question:
            self.status_var.set("Please enter a question first.")
            return
        try:
            top_k = int(self.top_k_var.get())
        except tk.TclError:
            self.status_var.set("Top K must be a whole number.")
            return
        top_k = max(1, top_k)
        self._set_running_state(True)
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(question, top_k, bool(self.show_passages_var.get())),
            daemon=True,
        )
        thread.start()

    def _run_pipeline(self, question: str, top_k: int, show_passages: bool) -> None:
        try:
            passages = self.pipeline.retrieve(question, top_k=top_k)
            answer = self.pipeline.generate(
                question,
                passages,
                reasoning_effort=self.reasoning_effort,
                text_verbosity=self.text_verbosity,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            display_passages: Sequence[RetrievedPassage] = passages if show_passages else []
            self.after(
                0, lambda: self._handle_result(answer.strip(), display_passages, error_message=None)
            )
        except Exception as exc:  # pragma: no cover - GUI side effect only
            self.after(0, lambda: self._handle_result("", [], error_message=str(exc)))

    def _handle_result(
        self,
        answer: str,
        passages: Sequence[RetrievedPassage],
        error_message: str | None,
    ) -> None:
        self._set_running_state(False)
        if error_message:
            self.status_var.set(f"Error: {error_message}")
            self._write_text(self.answer_text, error_message)
            self._write_text(self.passages_text, "")
            return
        formatted_answer = answer or "The model did not return any text."
        self._write_text(self.answer_text, formatted_answer)
        if passages:
            chunks = [
                f"[{idx + 1}] {p.reference} (score={p.score:.3f})\n{p.text}"
                for idx, p in enumerate(passages)
            ]
            passages_text = "\n\n".join(chunks)
        else:
            passages_text = "Passages are hidden. Enable the checkbox to view them."
        self._write_text(self.passages_text, passages_text)
        self.status_var.set("Ready.")

    def _set_running_state(self, running: bool) -> None:
        self.ask_button.config(state=tk.DISABLED if running else tk.NORMAL)
        if running:
            self.status_var.set("Working... this may take a moment.")

    def clear_fields(self) -> None:
        self.question_text.delete("1.0", tk.END)
        self._write_text(self.answer_text, "")
        self._write_text(self.passages_text, "")
        self.status_var.set("Cleared. Enter another question.")

    def _write_text(self, widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        if content:
            widget.insert(tk.END, content)
        widget.configure(state="disabled")
