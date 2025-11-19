#!/usr/bin/env python3
"""Command line utilities for building and querying a Bible RAG index."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import typer
from openai import OpenAI

DEFAULT_BIBLE_PATH = Path("data/kjv.json")
DEFAULT_INDEX_PATH = Path("artifacts/bible_index.npz")
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_GENERATOR_MODEL = "gpt-5.1"
PROMPT_TEMPLATE = (
    "You are a helpful assistant that answers questions about the Bible using only "
    "the supplied context. Cite the relevant passage references in your response.\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

FOOTNOTE_PATTERN = re.compile(r"\{[^}]+\}")
WHITESPACE_PATTERN = re.compile(r"\s+")
VALID_REASONING_EFFORTS = {"none", "low", "medium", "high"}
VALID_TEXT_VERBOSITIES = {"low", "medium", "high"}
EMBED_BACKENDS = {"auto", "openai", "sentence-transformers"}
GENERATOR_BACKENDS = {"auto", "openai", "huggingface"}

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
    embed_backend: str,
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
        "embed_backend": embed_backend,
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


def ensure_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export your OpenAI API key before running this command."
        )


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def validate_reasoning_effort(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized not in VALID_REASONING_EFFORTS:
        allowed = ", ".join(sorted(VALID_REASONING_EFFORTS))
        raise typer.BadParameter(f"Reasoning effort must be one of: {allowed}.")
    return normalized


def validate_text_verbosity(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized not in VALID_TEXT_VERBOSITIES:
        allowed = ", ".join(sorted(VALID_TEXT_VERBOSITIES))
        raise typer.BadParameter(f"Text verbosity must be one of: {allowed}.")
    return normalized


def validate_backend_option(value: str, option_name: str, allowed: Sequence[str]) -> str:
    normalized = (value or "auto").strip().lower()
    if not normalized:
        normalized = "auto"
    if normalized not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise typer.BadParameter(f"{option_name} must be one of: {allowed_values}.")
    return normalized


def infer_embed_backend(model_name: Optional[str]) -> str:
    if not model_name:
        return "openai"
    lowered = model_name.lower()
    if "/" in model_name or lowered.startswith("sentence-transformers"):
        return "sentence-transformers"
    return "openai"


def infer_generator_backend(model_name: Optional[str]) -> str:
    if not model_name:
        return "openai"
    lowered = model_name.lower()
    if lowered.startswith("gpt-"):
        return "openai"
    if "/" in model_name:
        return "huggingface"
    return "huggingface"


def resolve_embed_backend(selection: Optional[str], model_name: Optional[str]) -> str:
    normalized = (selection or "auto").strip().lower()
    if normalized == "auto" or not normalized:
        return infer_embed_backend(model_name)
    return normalized


def resolve_generator_backend(selection: Optional[str], model_name: Optional[str]) -> str:
    normalized = (selection or "auto").strip().lower()
    if normalized == "auto" or not normalized:
        return infer_generator_backend(model_name)
    return normalized


class OpenAIEmbedder:
    def __init__(self, model_name: str, client: Optional[OpenAI] = None) -> None:
        ensure_api_key()
        self.model_name = model_name
        self.client = client or OpenAI()

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            embeddings.extend([item.embedding for item in response.data])
        return normalize_vectors(np.array(embeddings, dtype=np.float32))


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is required for the 'sentence-transformers' backend."
            ) from exc
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        show_progress = len(texts) > batch_size
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)


class OpenAIResponsesGenerator:
    def __init__(self, model_name: str, client: Optional[OpenAI] = None) -> None:
        ensure_api_key()
        self.model_name = model_name
        self.client = client or OpenAI()

    def generate_text(
        self,
        prompt: str,
        *,
        reasoning_effort: str,
        text_verbosity: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            reasoning={"effort": reasoning_effort},
            text={"verbosity": text_verbosity},
            max_output_tokens=max_new_tokens,
        )
        return extract_response_text(response)


class HuggingFaceGenerator:
    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "torch and transformers are required for the 'huggingface' backend."
            ) from exc
        self.torch = torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        model_kwargs: Dict = {"low_cpu_mem_usage": True}
        if device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, **model_kwargs
        ).to(device)

    def generate_text(
        self,
        prompt: str,
        *,
        _reasoning_effort: str,
        _text_verbosity: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
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
        with self.torch.no_grad():
            output = self.generator.generate(**inputs, **generation_kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


def create_embedder(backend: str, model_name: str, client: Optional[OpenAI]) -> object:
    if backend == "openai":
        return OpenAIEmbedder(model_name, client=client)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unsupported embed backend: {backend}")


def create_generator(backend: str, model_name: str, client: Optional[OpenAI]) -> object:
    if backend == "openai":
        return OpenAIResponsesGenerator(model_name, client=client)
    if backend == "huggingface":
        return HuggingFaceGenerator(model_name)
    raise ValueError(f"Unsupported generator backend: {backend}")


def extract_response_text(response) -> str:
    chunks: List[str] = []
    for output in getattr(response, "output", []) or []:
        if getattr(output, "type", "") != "message":
            continue
        for content in getattr(output, "content", []) or []:
            if getattr(content, "type", "") == "output_text":
                chunks.append(getattr(content, "text", ""))
    if chunks:
        return "".join(chunks).strip()
    fallback = getattr(response, "output_text", None)
    if fallback:
        return str(fallback).strip()
    return "The model did not return any text."


class BibleRAGPipeline:
    def __init__(
        self,
        index_path: Path,
        embed_model_name: Optional[str] = None,
        generator_model: str = DEFAULT_GENERATOR_MODEL,
        embed_backend: Optional[str] = None,
        generator_backend: Optional[str] = None,
        client: Optional[OpenAI] = None,
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
        recorded_backend = meta.get("embed_backend")
        backend_choice = embed_backend or recorded_backend or "auto"
        self.embed_backend = resolve_embed_backend(backend_choice, self.embed_model_name)
        self.generator_model = generator_model
        self.generator_backend = resolve_generator_backend(generator_backend, self.generator_model)
        needs_openai = "openai" in {self.embed_backend, self.generator_backend}
        self.client = client
        if needs_openai and self.client is None:
            ensure_api_key()
            self.client = OpenAI()
        self.embedder = create_embedder(self.embed_backend, self.embed_model_name, client=self.client)
        self.generator = create_generator(self.generator_backend, self.generator_model, client=self.client)

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievedPassage]:
        if top_k <= 0:
            raise ValueError("top_k must be at least 1.")
        question_emb = self.embedder.embed_texts([question], batch_size=1)
        if question_emb.size == 0:
            return []
        question_emb = question_emb[0]
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
        reasoning_effort: str = "none",
        text_verbosity: str = "medium",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        if not passages:
            return "I could not find any relevant passages in the index."
        context_lines = [
            f"[{idx + 1}] {p.reference}: {p.text}" for idx, p in enumerate(passages)
        ]
        prompt = PROMPT_TEMPLATE.format(
            context="\n".join(context_lines), question=question.strip()
        )
        return self.generator.generate_text(
            prompt,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )


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
        help="Embedding model name (OpenAI or SentenceTransformer).",
    ),
    batch_size: int = typer.Option(32, help="Number of passages per embedding API request."),
    embed_backend: str = typer.Option(
        "auto",
        help="Embedding backend to use (auto, openai, sentence-transformers).",
    ),
) -> None:
    """Create an embedding index for the supplied Bible corpus."""

    validate_chunk_config(chunk_size, chunk_overlap)
    embed_backend = validate_backend_option(embed_backend, "embed-backend", EMBED_BACKENDS)
    resolved_backend = resolve_embed_backend(embed_backend, embed_model)
    books = load_books(bible_path)
    passages = list(iter_passages(books, chunk_size, chunk_overlap))
    if not passages:
        typer.secho("No passages were produced. Check the input file.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho(f"Loaded {len(passages)} passages. Encoding...", fg=typer.colors.GREEN)
    client: Optional[OpenAI] = None
    if resolved_backend == "openai":
        ensure_api_key()
        client = OpenAI()
    embedder = create_embedder(resolved_backend, embed_model, client=client)
    embeddings = embedder.embed_texts([p.text for p in passages], batch_size=batch_size)
    save_index(
        passages,
        embeddings,
        output_path,
        embed_model,
        resolved_backend,
        chunk_size,
        chunk_overlap,
        bible_path,
    )
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
        DEFAULT_GENERATOR_MODEL,
        help="Generator model name (OpenAI Responses or HuggingFace seq2seq).",
    ),
    embed_model: Optional[str] = typer.Option(
        None, help="Override the embedding model if it differs from the index metadata."
    ),
    reasoning_effort: str = typer.Option(
        "none",
        help="Reasoning budget to pass to GPT-5.1 (none, low, medium, high).",
        callback=validate_reasoning_effort,
    ),
    text_verbosity: str = typer.Option(
        "medium",
        help="Controls answer verbosity (low, medium, high).",
        callback=validate_text_verbosity,
    ),
    max_new_tokens: int = typer.Option(256, help="Maximum tokens to generate."),
    temperature: float = typer.Option(
        0.0,
        help="Sampling temperature (used only for HuggingFace generator backends).",
    ),
    embed_backend: str = typer.Option(
        "auto",
        help="Embedding backend to use when asking (auto, openai, sentence-transformers).",
    ),
    generator_backend: str = typer.Option(
        "auto",
        help="Generator backend to use when asking (auto, openai, huggingface).",
    ),
    show_passages: bool = typer.Option(
        True, help="Display the retrieved passages along with the answer."
    ),
) -> None:
    """Answer a question using retrieval-augmented generation over the Bible."""

    embed_backend = validate_backend_option(embed_backend, "embed-backend", EMBED_BACKENDS)
    generator_backend = validate_backend_option(
        generator_backend, "generator-backend", GENERATOR_BACKENDS
    )
    engine = BibleRAGPipeline(
        index_path,
        embed_model_name=embed_model,
        generator_model=generator_model,
        embed_backend=embed_backend,
        generator_backend=generator_backend,
    )
    retrieved = engine.retrieve(question, top_k=top_k)
    answer = engine.generate(
        question,
        retrieved,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    typer.echo("Answer:\n" + answer)
    if engine.generator_backend == "huggingface" and (
        reasoning_effort != "none" or text_verbosity != "medium"
    ):
        typer.secho(
            "Note: reasoning_effort and text_verbosity apply only to OpenAI generators.",
            fg=typer.colors.YELLOW,
        )
    if engine.generator_backend == "openai" and temperature > 0:
        typer.secho(
            "Note: temperature is ignored when using OpenAI Responses models.",
            fg=typer.colors.YELLOW,
        )
    if show_passages and retrieved:
        typer.echo("\nTop passages:")
        for idx, passage in enumerate(retrieved, start=1):
            typer.echo(
                f"[{idx}] {passage.reference} (score={passage.score:.3f})\n{passage.text}\n"
            )


if __name__ == "__main__":
    app()
