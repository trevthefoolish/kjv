# kjv

Do this in remembrance of me..

This repository contains a minimal Retrieval-Augmented Generation (RAG) workflow for answering questions about the Bible (King James Version). It ingests the public-domain text, chunks it into overlapping passages, indexes those passages with sentence-transformer embeddings, and uses a local seq2seq model (FLAN-T5) to generate answers grounded in the retrieved context.

## Prerequisites

- Python 3.9+
- (Optional but recommended) A virtual environment to isolate dependencies
- Enough disk space for HuggingFace models (~1.5 GB for FLAN-T5-base + embeddings)

Install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> The first run will download the embedding and generation models from HuggingFace. They are cached afterwards.

## Build the embedding index

The repository already contains `data/kjv.json` (downloaded from [thiagobodruk/bible](https://github.com/thiagobodruk/bible)). To chunk the text and build the embedding index run:

```bash
python bible_rag.py build-index \
  --bible-path data/kjv.json \
  --output-path artifacts/bible_index.npz \
  --chunk-size 6 \
  --chunk-overlap 2 \
  --embed-model sentence-transformers/all-MiniLM-L6-v2
```

- `chunk-size` controls how many verses are merged into one passage.
- `chunk-overlap` lets neighboring passages share verses for better recall.
- `embed-model` can be replaced with any `SentenceTransformer` checkpoint.

The command stores the vectors plus metadata under `artifacts/` and records the configuration in `artifacts/bible_index.meta.json`.

## Ask questions with RAG

Once the index is available you can answer questions:

```bash
python bible_rag.py ask "Who led the Israelites through the Red Sea?"
```

By default this will:

- Retrieve the top 5 passages using the embeddings saved in the index.
- Feed the passages and the question into `google/flan-t5-base` to generate an answer.
- Display the retrieved passages (set `--show-passages False` to hide them).

Useful flags:

- `--top-k 8` – change the number of retrieved passages.
- `--generator-model google/flan-t5-large` – swap the language model.
- `--temperature 0.7` – enable sampling for more diverse answers.
- `--max-new-tokens 128` – control the answer length.

If you change the embedding model when asking (`--embed-model`), make sure it matches the model used for the index to avoid cosine similarity mismatches.

## How it works

1. `build-index` loads the JSON Bible, removes inline footnotes, chunks verses, and embeds each chunk with the specified SentenceTransformer.
2. The embeddings, text, and metadata are persisted in `NPZ` form for quick loading.
3. `ask` loads the vectors, retrieves the most relevant passages with cosine similarity, builds a context prompt, and lets FLAN-T5 generate an answer that cites the retrieved references.

Feel free to extend this by swapping in other corpora, embedding models, chunking strategies, or vector stores.
