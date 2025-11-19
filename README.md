# kjv

Do this in remembrance of me..

This repository contains a minimal Retrieval-Augmented Generation (RAG) workflow for answering questions about the Bible (King James Version). It ingests the public-domain text, chunks it into overlapping passages, indexes those passages, and feeds the results into a generator. By default it leans on OpenAI's `text-embedding-3-large` + [`gpt-5.1` Responses](https://platform.openai.com/docs/guides/latest-model) stack, but a single flag switches everything back to the original HuggingFace SentenceTransformer + FLAN-T5 pipeline for an open, offline fallback.

## Prerequisites

- Python 3.9+
- (Optional but recommended) A virtual environment to isolate dependencies
- An [OpenAI API key](https://platform.openai.com/account/api-keys) exported as `OPENAI_API_KEY` when using the OpenAI backend (the HuggingFace fallback runs locally and does not require it).

Install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> The CLI can either hit the OpenAI API or stay completely local. Set `OPENAI_API_KEY` only if you plan to use the `openai` embedding or generator backends. The HuggingFace fallback will download the relevant models the first time you run it.

## Build the embedding index

The repository already contains `data/kjv.json` (downloaded from [thiagobodruk/bible](https://github.com/thiagobodruk/bible)). To chunk the text and build the embedding index run:

```bash
python bible_rag.py build-index \
  --bible-path data/kjv.json \
  --output-path artifacts/bible_index.npz \
  --chunk-size 6 \
  --chunk-overlap 2 \
  --embed-model text-embedding-3-large
```

- `chunk-size` controls how many verses are merged into one passage.
- `chunk-overlap` lets neighboring passages share verses for better recall.
- `embed-model` can be swapped for any OpenAI embedding checkpoint (see the [`text-embedding-3-large` docs](https://platform.openai.com/docs/models/text-embedding-3-large)).
- Pass `--embed-backend sentence-transformers --embed-model sentence-transformers/all-MiniLM-L6-v2` to recreate the fully local pipeline.

> A ready-to-use index generated with `text-embedding-3-large` ships in `artifacts/bible_index.npz` (with metadata beside it), so you only need to run `build-index` if you want to experiment with different chunking or model settings. The metadata also records the backend that produced the embeddings so the `ask` command can stay in sync automatically.

The command stores the vectors plus metadata under `artifacts/` and records the configuration in `artifacts/bible_index.meta.json`.

## Ask questions with RAG

Once the index is available you can answer questions:

```bash
python bible_rag.py ask "Who led the Israelites through the Red Sea?"
```

By default this will:

- Retrieve the top 5 passages using the embeddings saved in the index.
- Feed the passages and the question into OpenAI's [`gpt-5.1` Responses model](https://platform.openai.com/docs/guides/latest-model) to generate an answer with reasoning effort `none` and verbosity `medium`.
- Display the retrieved passages (set `--show-passages False` to hide them).

To keep everything local, add:

```bash
python bible_rag.py ask "Who led the Israelites through the Red Sea?" \
  --embed-backend sentence-transformers \
  --embed-model sentence-transformers/all-MiniLM-L6-v2 \
  --generator-backend huggingface \
  --generator-model google/flan-t5-base \
  --temperature 0.3
```

Useful flags:

- `--top-k 8` – change the number of retrieved passages.
- `--generator-model gpt-5-mini` – pick a cheaper GPT-5 family model (e.g., `gpt-5-mini`, `gpt-5-nano`).
- `--generator-backend huggingface` – switch to seq2seq models like `google/flan-t5-base` for an offline run (pair with `--temperature` if you want sampling).
- `--reasoning-effort high` – allow GPT-5.1 to spend more chain-of-thought tokens on complex questions (choices: none/low/medium/high).
- `--text-verbosity low` – ask GPT-5.1 to respond more concisely (choices: low/medium/high).
- `--temperature 0.7` – enable sampling (huggingface generators only).
- `--embed-backend sentence-transformers` – keep using the embeddings that were baked into the index even if they came from SentenceTransformers.
- `--max-new-tokens 128` – control the answer length.

If you change the embedding model when asking (`--embed-model`), make sure it matches the model used for the index (or override the backend) to avoid cosine similarity mismatches. The GPT-5 family does **not** accept `temperature`, `top_p`, or `logprobs` parameters; use `--reasoning-effort`, `--text-verbosity`, and `--max-new-tokens` to steer Responses output instead. The huggingface generators ignore those knobs but honor `--temperature`. Only the OpenAI workflows require an API key or incur API usage.

## How it works

1. `build-index` loads the JSON Bible, removes inline footnotes, chunks verses, and embeds each chunk with whichever backend/model pair you selected.
2. The embeddings, text, and metadata (including model + backend) are persisted in `NPZ` form for quick loading.
3. `ask` loads the vectors, retrieves the most relevant passages with cosine similarity, builds a context prompt, and feeds that to either OpenAI's Responses API or a local seq2seq generator to produce citations.

Feel free to extend this by swapping in other corpora, embedding models, chunking strategies, or vector stores.
