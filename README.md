# ReportIQ — Financial & Business Report Q&A (Multimodal RAG)

Turn dense PDF reports — quarterly financials, board decks, procurement
statements — into a chat interface, including the tables and charts that
plain-text tools choke on. Built entirely on free/open-source components,
so there's no per-seat licensing cost to pass on to a client.

**Live in 60 seconds:** the app ships with a synthetic sample report
(`sample_documents/demo_quarterly_business_report.pdf` — revenue table with
merged/nested headers, a bar chart, a pie chart) preloaded automatically on
first run, so anyone opening the app can immediately ask "What was EMEA's
Q3 revenue?" or "What does the regional pie chart show?" without uploading
anything.

## Why this matters for clients

Finance, ops, and PE/VC teams routinely pay someone to manually re-key
numbers out of PDF reports into spreadsheets, or to hunt through 40-page
board decks for one figure. This does that lookup instantly and cites the
exact page/table/chart it pulled the answer from, so the answer is
verifiable, not a black box.

## Suggested Upwork positioning

- **Gig title angle:** "AI assistant that answers questions from your
  financial reports, invoices, or business PDFs — including tables and
  charts."
- **Demo flow on a call:** open the live link, ask 2-3 questions against
  the preloaded sample report, then upload the client's own PDF live and
  ask a question about a number that's actually in one of their tables.
  That's the moment that closes the deal — it's not a canned demo.
- **Deploy for a live link** (free): [Streamlit Community Cloud](https://streamlit.io/cloud) —
  push this repo to GitHub, connect it, set `GROQ_API_KEY` as a secret, and
  you get a shareable `*.streamlit.app` URL to put directly in proposals
  instead of "clone this repo."
- Keep `sample_documents/demo_quarterly_business_report.pdf` in the repo
  even after deploying — it's what makes the live link demoable without a
  client needing to prep a file first.

## Architecture

```
                     ┌─────────────────────┐
        PDF  ───────▶│  document_parser.py │  unstructured (hi_res)
                     │  - text elements     │  infer_table_structure=True
                     │  - table HTML        │
                     │  - extracted images  │
                     └─────────┬───────────┘
                               │ scanned pages (~0 text)
                               ▼
                     ┌─────────────────────┐
                     │    ocr_engine.py     │  pytesseract / paddleocr
                     │  full-page OCR       │
                     └─────────┬───────────┘
                               │
     chart/figure images ─────┼──────────────┐
                               ▼              ▼
                     ┌─────────────────────┐  ┌──────────────────┐
                     │  chart_analyzer.py  │  │ table HTML -> DF │
                     │  BLIP caption + OCR │  │ (pandas)         │
                     │  + chart-type guess │  └──────────────────┘
                     └─────────┬───────────┘
                               ▼
                     ┌─────────────────────┐
                     │    embeddings.py     │  MiniLM (unified text space)
                     └─────────┬───────────┘
                               ▼
                     ┌─────────────────────┐
                     │   vector_store.py    │  ChromaDB (persistent)
                     └─────────┬───────────┘
                               ▲  query embedding
                               │
Question ──▶ rag_pipeline.py ──┴──▶ retrieve top-k ──▶ llm_client.py (Groq / HF)
                                                            │
                                                            ▼
                                                     Grounded answer + sources
```

**Key design choice:** tables and charts are converted into rich, faithful
*text* (markdown tables from preserved HTML; chart descriptions from
caption + OCR + heuristics) and embedded in the **same** MiniLM space as
narrative text. This lets one retrieval call surface text, tables, and
charts together, ranked by relevance, and lets a normal text LLM reason
over all of them without needing a multimodal LLM at inference time. A
secondary CLIP index (`images` collection) is wired up for optional
visual-similarity search but isn't required for Q&A.

### Retrieval: hybrid search + reranking

Pure dense search misses exact tokens that matter most in tables/charts -
specific numbers, codes, dates. `src/retrieval.py` adds:

1. **BM25 keyword search** over the same corpus, run alongside the dense
   (MiniLM) search.
2. **Reciprocal rank fusion** to merge the two rankings without needing to
   normalize incomparable score scales.
3. **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over
   the fused candidate pool, which is what actually improves final
   precision - fusion gets a good pool, the reranker picks the best few.

Toggle either independently via `ENABLE_HYBRID_SEARCH` / `ENABLE_RERANKING`
in `.env` - both degrade gracefully to plain dense top-k if disabled.

### Ingestion dedup

Each ingested file's SHA-256 hash is recorded in `data/ingest_manifest.json`.
Re-uploading an identical PDF (or restarting the app and re-ingesting) is a
no-op instead of reprocessing and re-embedding everything again.

## Setup

### System dependencies (required for parsing/OCR)
```bash
# Debian/Ubuntu
sudo apt-get install -y poppler-utils tesseract-ocr libgl1
```
`poppler-utils` is needed by `pdf2image` (scanned-page rasterization) and by
`unstructured`'s PDF backend. `tesseract-ocr` is needed by `pytesseract`.

### Python dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`unstructured`'s `hi_res` strategy downloads a small layout-detection model
on first use. If you hit install/resource issues with `hi_res` (e.g. on a
low-memory machine), set:
```bash
export UNSTRUCTURED_STRATEGY=fast   # text only, no table structure
# or
export UNSTRUCTURED_STRATEGY=ocr_only
```

### API keys (generation backend — both free)
```bash
export GROQ_API_KEY=your_key       # https://console.groq.com (preferred: fast, generous free tier)
export HF_API_TOKEN=your_token     # https://huggingface.co/settings/tokens (fallback)
```
Or copy `.env.example` to `.env` and fill it in (`python-dotenv` is already
in `requirements.txt` if you want to auto-load it).

With neither set, the app still ingests and retrieves; it just can't
generate a final natural-language answer (it will say so explicitly).

### Optional: swap in PaddleOCR for better scanned-page accuracy
```bash
pip install paddleocr paddlepaddle
export OCR_ENGINE=paddleocr
```

## Run
```bash
streamlit run app.py
```
1. Upload one or more PDFs in the sidebar and click **Ingest uploaded PDFs**.
2. Watch the per-file breakdown (text / table / chart / OCR-recovered chunk
   counts).
3. Ask questions in the chat box. Each answer expands into its retrieved
   sources — rendered as a table, an image with its generated description,
   or a text snippet — with page numbers and similarity scores, so you can
   verify every claim.

## Known limitations (be upfront about these)

- **No open model reads exact numeric values off a chart reliably.**
  `chart_analyzer.py` combines a caption model + OCR (which *does* pick up
  literal printed axis labels/legends/numbers) + a coarse chart-type
  heuristic. This is good enough for retrieval ("find the chart about
  Q3 revenue") and for approximate descriptions, not for precise value
  extraction from a bar's height. The system prompt tells the LLM to flag
  this uncertainty rather than fabricate exact figures.
- `hi_res` table detection is good but not perfect on very dense, nested
  multi-index tables — always check the rendered table in the "Sources"
  panel against the answer.
- First run downloads several model weights (BLIP, MiniLM, CLIP) — expect
  a delay and make sure the machine has network + disk for that.
- This is a single-process demo app (Chroma persistent client, in-memory
  chat history) — for multi-user production use, put Chroma behind its
  server mode and move chat history to a real session store.

## Recommended next steps (not yet implemented)

Roughly in priority order for taking this from demo to production:

- **Streaming responses** — Groq's API supports streaming; wiring that into
  `llm_client.py` + `st.write_stream` would remove the "wait for the full
  answer" delay in the UI.
- **Retrieval evaluation harness** — a small labeled Q&A set per document
  plus a free eval framework (e.g. RAGAS) to track precision/recall as you
  tune chunk size, top_k, or swap models, instead of eyeballing answers.
- **Async/background ingestion** — large PDFs currently block the Streamlit
  thread during parsing; a background worker (e.g. `concurrent.futures` or
  a task queue) with a progress callback would keep the UI responsive.
- **Table-aware chunking for very large tables** — a table with hundreds of
  rows currently becomes one large markdown blob; splitting by logical row
  groups (with the header repeated in each chunk) would improve retrieval
  precision on big tables specifically.
- **Query rewriting / HyDE** for vague or multi-part questions before
  retrieval, to improve recall on questions that don't lexically match the
  source text.
- **Dockerfile** for one-command deployment, since the system dependency
  list (poppler, tesseract, libgl1) is easy to get wrong by hand.
- **Multi-format ingestion** (docx, pptx, xlsx) via `unstructured`'s other
  partitioners, if the use case grows beyond PDFs.
- **Chroma server mode** instead of the embedded persistent client, if this
  needs to support concurrent multi-user access.

## Project layout
```
multimodal_rag/
├── app.py                 # Streamlit UI (branded as "ReportIQ")
├── config.py              # all tunables (models, paths, thresholds)
├── requirements.txt
├── .env.example
├── sample_documents/
│   └── demo_quarterly_business_report.pdf   # auto-loaded demo doc
├── scripts/
│   └── generate_sample_pdf.py               # regenerate/customize the demo doc
├── src/
│   ├── document_parser.py # unstructured-based PDF parsing -> Chunks
│   ├── ocr_engine.py      # scanned-page OCR fallback
│   ├── chart_analyzer.py  # caption + OCR + heuristic chart-type
│   ├── embeddings.py      # MiniLM (text) + CLIP (image) embedders
│   ├── vector_store.py    # ChromaDB wrapper
│   ├── retrieval.py       # BM25 + reciprocal rank fusion + cross-encoder reranking
│   ├── llm_client.py      # Groq + HF Inference API generation
│   └── rag_pipeline.py    # orchestration: ingest() / answer()
└── data/                  # uploads, extracted images, chroma_db (gitignored)
```
