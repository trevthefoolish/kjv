#!/usr/bin/env python3
"""Command line utilities for building and querying a Bible RAG index."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import typer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_BIBLE_PATH = Path("data/kjv.json")
DEFAULT_INDEX_PATH = Path("artifacts/bible_index.npz")
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GENERATOR_MODEL = "google/flan-t5-base"
PROMPT_TEMPLATE = (
    "You are a helpful assistant that answers questions about the Bible using only "
    "the supplied context. Cite the relevant passage references in your response.\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

FOOTNOTE_PATTERN = re.compile(r"\{[^}]+\}")
WHITESPACE_PATTERN = re.compile(r"\s+")

app = typer.Typer(help="Build an embedding index for the Bible and run retrieval augmented generation over it.")


@dataclass
class Passage:
    """Represents a contiguous chunk of verses."""

    text: str
    reference: str
    book: str
    chapter: int
    verse_start: int
    verse_end: int


@dataclass
class RetrievedPassage(Passage):
    score: float


def clean_verse(raw_text: str) -> str:
    """Remove inline commentary and collapse whitespace."""

    without_notes = FOOTNOTE_PATTERN.sub("", raw_text)
    return WHITESPACE_PATTERN.sub(" ", without_notes).strip()


def load_books(bible_path: Path) -> List[Dict]:
    if not bible_path.exists():
        raise FileNotFoundError(f"Bible file not found: {bible_path}")
    with bible_path.open(encoding="utf-8-sig") as source:
        return json.load(source)


def validate_chunk_config(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise typer.BadParameter("Chunk size must be greater than zero.")
    if chunk_overlap < 0:
        raise typer.BadParameter("Chunk overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise typer.BadParameter("Chunk overlap must be smaller than the chunk size.")


def iter_passages(books: Sequence[Dict], chunk_size: int, chunk_overlap: int) -> Iterable[Passage]:
    step = chunk_size - chunk_overlap
    for book in books:
        book_name = book.get("name", book.get("abbrev", ""))
        for chapter_idx, verses in enumerate(book.get("chapters", []), start=1):
            cleaned = [clean_verse(v) for v in verses]
            cleaned = [v for v in cleaned if v]
            if not cleaned:
                continue
            for start in range(0, len(cleaned), step):
                end = min(start + chunk_size, len(cleaned))
                chunk = " ".join(cleaned[start:end]).strip()
                if not chunk:
                    continue
                verse_start = start + 1
                verse_end = end
                if verse_start == verse_end:
                    reference = f"{book_name} {chapter_idx}:{verse_start}"
                else:
                    reference = f"{book_name} {chapter_idx}:{verse_start}-{verse_end}"
                yield Passage(
                    text=chunk,
                    reference=reference,
                    book=book_name,
                    chapter=chapter_idx,
                    verse_start=verse_start,
                    verse_end=verse_end,
                )


def save_index(
    passages: Sequence[Passage],
    embeddings: np.ndarray,
    output_path: Path,
    embed_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    bible_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings.astype(np.float32),
        texts=np.array([p.text for p in passages], dtype=object),
        references=np.array([p.reference for p in passages], dtype=object),
        books=np.array([p.book for p in passages], dtype=object),
        chapters=np.array([p.chapter for p in passages], dtype=np.int32),
        verse_starts=np.array([p.verse_start for p in passages], dtype=np.int32),
        verse_ends=np.array([p.verse_end for p in passages], dtype=np.int32),
    )
    meta = {
        "embed_model": embed_model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_passages": len(passages),
        "bible_path": str(bible_path),
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_index(index_path: Path) -> Dict[str, np.ndarray]:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found at {index_path}. Run the build-index command first."
        )
    data = np.load(index_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_metadata(index_path: Path) -> Dict[str, str]:
    meta_path = index_path.with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


class BibleRAGPipeline:
    def __init__(
        self,
        index_path: Path,
        embed_model_name: Optional[str] = None,
        generator_model: str = DEFAULT_GENERATOR_MODEL,
    ) -> None:
        self.index_path = index_path
        arrays = load_index(index_path)
        self.embeddings = arrays["embeddings"]
        self.texts = arrays["texts"]
        self.references = arrays["references"]
        self.books = arrays.get("books")
        self.chapters = arrays.get("chapters")
        self.verse_starts = arrays.get("verse_starts")
        self.verse_ends = arrays.get("verse_ends")
        meta = load_metadata(index_path)
        self.embed_model_name = embed_model_name or meta.get("embed_model", DEFAULT_EMBED_MODEL)
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs: Dict = {"low_cpu_mem_usage": True}
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(
            generator_model, **model_kwargs
        ).to(self.device)

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievedPassage]:
        if top_k <= 0:
            raise ValueError("top_k must be at least 1.")
        question_emb = self.embedder.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores = self.embeddings @ question_emb
        top_k = min(top_k, len(scores))
        if top_k == 0:
            return []
        best_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        best_indices = best_indices[np.argsort(-scores[best_indices])]
        results = []
        for idx in best_indices:
            book = str(self.books[idx]) if self.books is not None else ""
            chapter = int(self.chapters[idx]) if self.chapters is not None else 0
            verse_start = int(self.verse_starts[idx]) if self.verse_starts is not None else 0
            verse_end = int(self.verse_ends[idx]) if self.verse_ends is not None else 0
            results.append(
                RetrievedPassage(
                    text=str(self.texts[idx]),
                    reference=str(self.references[idx]),
                    book=book,
                    chapter=chapter,
                    verse_start=verse_start,
                    verse_end=verse_end,
                    score=float(scores[idx]),
                )
            )
        return results

    def generate(
        self,
        question: str,
        passages: Sequence[RetrievedPassage],
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> str:
        if not passages:
            return "I could not find any relevant passages in the index."
        context_lines = [
            f"[{idx + 1}] {p.reference}: {p.text}" for idx, p in enumerate(passages)
        ]
        prompt = PROMPT_TEMPLATE.format(
            context="\n".join(context_lines), question=question.strip()
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs.update(temperature=temperature, top_p=0.9)
        with torch.no_grad():
            output = self.generator.generate(**inputs, **generation_kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


@app.command("build-index")
def build_index_command(
    bible_path: Path = typer.Option(
        DEFAULT_BIBLE_PATH,
        help="Path to the JSON file containing the Bible text (default: data/kjv.json).",
    ),
    output_path: Path = typer.Option(
        DEFAULT_INDEX_PATH,
        help="Where to store the generated embedding index (NPZ file).",
    ),
    chunk_size: int = typer.Option(6, help="Number of verses per chunk."),
    chunk_overlap: int = typer.Option(2, help="Number of verses to overlap between chunks."),
    embed_model: str = typer.Option(
        DEFAULT_EMBED_MODEL,
        help="SentenceTransformer model name for embeddings.",
    ),
    batch_size: int = typer.Option(32, help="Embedding batch size."),
) -> None:
    """Create an embedding index for the supplied Bible corpus."""

    validate_chunk_config(chunk_size, chunk_overlap)
    books = load_books(bible_path)
    passages = list(iter_passages(books, chunk_size, chunk_overlap))
    if not passages:
        typer.secho("No passages were produced. Check the input file.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho(f"Loaded {len(passages)} passages. Encoding...", fg=typer.colors.GREEN)
    model = SentenceTransformer(embed_model)
    embeddings = model.encode(
        [p.text for p in passages],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    save_index(passages, embeddings, output_path, embed_model, chunk_size, chunk_overlap, bible_path)
    typer.secho(
        f"Index saved to {output_path} ({len(passages)} passages).",
        fg=typer.colors.GREEN,
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about the Bible."),
    index_path: Path = typer.Option(DEFAULT_INDEX_PATH, help="Path to the index NPZ file."),
    top_k: int = typer.Option(5, help="Number of passages to retrieve.", min=1),
    generator_model: str = typer.Option(
        DEFAULT_GENERATOR_MODEL, help="Seq2Seq model used for answer generation."
    ),
    embed_model: Optional[str] = typer.Option(
        None, help="Override the embedding model if it differs from the index metadata."
    ),
    temperature: float = typer.Option(0.0, help="Sampling temperature for generation."),
    max_new_tokens: int = typer.Option(256, help="Maximum tokens to generate."),
    show_passages: bool = typer.Option(
        True, help="Display the retrieved passages along with the answer."
    ),
) -> None:
    """Answer a question using retrieval-augmented generation over the Bible."""

    engine = BibleRAGPipeline(index_path, embed_model_name=embed_model, generator_model=generator_model)
    retrieved = engine.retrieve(question, top_k=top_k)
    answer = engine.generate(question, retrieved, temperature=temperature, max_new_tokens=max_new_tokens)
    typer.echo("Answer:\n" + answer)
    if show_passages and retrieved:
        typer.echo("\nTop passages:")
        for idx, passage in enumerate(retrieved, start=1):
            typer.echo(
                f"[{idx}] {passage.reference} (score={passage.score:.3f})\n{passage.text}\n"
            )


if __name__ == "__main__":
    app()
