import numpy as np
import pytest
import typer
from pathlib import Path
from types import SimpleNamespace

import bible_rag
from bible_rag import (
    BibleRAGPipeline,
    Passage,
    RetrievedPassage,
    clean_verse,
    iter_passages,
    load_index,
    load_metadata,
    save_index,
    validate_chunk_config,
)


def build_sample_index(tmp_path):
    """Create a lightweight index on disk for pipeline tests."""

    passages = [
        Passage("In the beginning God created the heaven and the earth.", "Genesis 1:1", "Genesis", 1, 1, 1),
        Passage("And the earth was without form, and void; and darkness was upon the face of the deep.", "Genesis 1:2", "Genesis", 1, 2, 2),
        Passage("And the Spirit of God moved upon the face of the waters.", "Genesis 1:3", "Genesis", 1, 3, 3),
    ]
    embeddings = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.2, 0.8, 0.1],
            [-0.5, 0.0, 0.9],
        ],
        dtype=np.float32,
    )
    output_path = tmp_path / "stub_index.npz"
    save_index(
        passages,
        embeddings,
        output_path,
        embed_model_name="stub-embedding",
        embed_backend="openai",
        chunk_size=2,
        chunk_overlap=1,
        bible_path=Path("data/kjv.json"),
    )
    return output_path, passages, embeddings


def test_clean_verse_removes_footnotes_and_whitespace():
    raw = "In the beginning {footnote}  God created \n the heaven."
    assert clean_verse(raw) == "In the beginning God created the heaven."


def test_iter_passages_yields_overlapping_chunks():
    books = [
        {
            "name": "Genesis",
            "chapters": [
                [
                    "Verse 1",
                    "Verse 2",
                    "{comment} Verse 3",
                ]
            ],
        }
    ]
    passages = list(iter_passages(books, chunk_size=2, chunk_overlap=1))
    references = [p.reference for p in passages]
    assert references == ["Genesis 1:1-2", "Genesis 1:2-3", "Genesis 1:3"]


def test_validate_chunk_config_rejects_invalid_values():
    with pytest.raises(typer.BadParameter):
        validate_chunk_config(0, 0)

    with pytest.raises(typer.BadParameter):
        validate_chunk_config(2, 3)


def test_save_and_load_index_round_trip(tmp_path):
    index_path, passages, embeddings = build_sample_index(tmp_path)
    arrays = load_index(index_path)
    assert np.allclose(arrays["embeddings"], embeddings.astype(np.float32))
    assert list(arrays["references"]) == [p.reference for p in passages]

    meta = load_metadata(index_path)
    assert meta["embed_model"] == "stub-embedding"
    assert meta["embed_backend"] == "openai"
    assert meta["num_passages"] == len(passages)


def test_pipeline_retrieve_uses_cosine_scores(tmp_path, monkeypatch):
    index_path, passages, _ = build_sample_index(tmp_path)

    class StubEmbedder:
        def __init__(self):
            self.embedding = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            self.requests = []

        def embed_texts(self, texts, batch_size=32):
            self.requests.append(list(texts))
            return self.embedding

    stub_embedder = StubEmbedder()
    monkeypatch.setattr(
        bible_rag, "create_embedder", lambda *args, **kwargs: stub_embedder
    )

    class StubGenerator:
        def generate_text(self, *args, **kwargs):
            return "unused"

    monkeypatch.setattr(
        bible_rag, "create_generator", lambda *args, **kwargs: StubGenerator()
    )

    pipeline = BibleRAGPipeline(index_path=index_path, client=SimpleNamespace())
    results = pipeline.retrieve("Let there be light", top_k=2)

    assert [r.reference for r in results] == [passages[0].reference, passages[1].reference]
    assert stub_embedder.requests == [["Let there be light"]]


def test_pipeline_generate_formats_context_and_returns_text(tmp_path, monkeypatch):
    index_path, passages, _ = build_sample_index(tmp_path)

    class StubEmbedder:
        def embed_texts(self, texts, batch_size=32):
            return np.zeros((1, 3), dtype=np.float32)

    monkeypatch.setattr(
        bible_rag, "create_embedder", lambda *args, **kwargs: StubEmbedder()
    )

    class RecordingGenerator:
        def __init__(self, payload="Mock answer"):
            self.payload = payload
            self.calls = []

        def generate_text(
            self,
            prompt,
            *,
            reasoning_effort: str,
            text_verbosity: str,
            max_new_tokens: int,
            temperature: float,
        ) -> str:
            self.calls.append(
                {
                    "prompt": prompt,
                    "reasoning_effort": reasoning_effort,
                    "text_verbosity": text_verbosity,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                }
            )
            return self.payload

    generator = RecordingGenerator()
    monkeypatch.setattr(
        bible_rag, "create_generator", lambda *args, **kwargs: generator
    )

    pipeline = BibleRAGPipeline(index_path=index_path, client=SimpleNamespace())
    retrieved = [
        RetrievedPassage(
            text=passages[0].text,
            reference=passages[0].reference,
            book=passages[0].book,
            chapter=passages[0].chapter,
            verse_start=passages[0].verse_start,
            verse_end=passages[0].verse_end,
            score=0.95,
        )
    ]

    answer = pipeline.generate(
        "Who created the heaven and the earth?",
        retrieved,
        reasoning_effort="medium",
        text_verbosity="low",
        max_new_tokens=64,
        temperature=0.0,
    )

    assert answer == "Mock answer"
    assert generator.calls
    recorded = generator.calls[0]
    assert recorded["reasoning_effort"] == "medium"
    assert "[1] Genesis 1:1" in recorded["prompt"]
    assert "Who created the heaven and the earth?" in recorded["prompt"]
