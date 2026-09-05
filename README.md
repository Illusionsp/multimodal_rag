# ReportIQ

**A production-ready multimodal RAG assistant for PDF reports.**

ReportIQ turns dense PDFs — financial reports, board decks, procurement
statements — into a chat interface. Upload a document with complex tables,
charts, and scanned pages, and get instant, hallucination-resistant answers
with exact source citations. Built entirely on free/open-source components.

## 🚀 Key Features

- **Multimodal Parsing**: Extracts text, complex tables (merged cells,
  multi-level headers), and charts/infographics from PDFs — including
  scanned pages via OCR fallback.
- **Hybrid Search**: Combines BM25 keyword search with semantic embeddings
  for high-recall retrieval across text, tables, and charts.
- **Cross-Encoder Reranking**: Re-ranks the top candidates down to the
  most relevant few using a cross-encoder model for higher accuracy.
- **Strict Grounding**: The LLM is prompted to answer only from retrieved
  context, and cites the exact page/table/chart behind every answer.
- **Streamlit Interface**: A native Python UI for uploading documents and
  asking questions.
- **Production-Ready**: Containerized with Docker, with an offline test
  suite covering core logic.
- **Zero-Setup Demo**: Ships with a sample financial report so it's
  testable immediately, with no upload required.

## 🧠 System Architecture

```text
User Query
   │
   ▼
Query Embedding (sentence-transformers MiniLM)
   │
   ▼
Hybrid Retrieval (BM25 + Semantic Search)   ───► Top-N Candidates
   │
   ▼
Cross-Encoder Reranker (ms-marco-MiniLM)    ───► Top-K Chunks
   │
   ▼
LLM Generation (Groq / Hugging Face)        ───► Grounded Answer + Citations
```

Ingestion side (PDF → searchable index):

```text
PDF
   │
   ▼
Parsing (unstructured, hi_res)  ───► text · table HTML · extracted images
   │
   ├─► Scanned pages  ──► OCR fallback (pytesseract / paddleocr)
   └─► Charts/images   ──► Caption (BLIP) + OCR + chart-type heuristic
   │
   ▼
Unified text embedding (MiniLM)  ───► ChromaDB (persistent vector store)
```

Text, tables, and charts all land in the same embedding space, so one
query retrieves across all three, ranked together.

## 🛠️ Technology Stack

- **Frontend/UI**: Streamlit
- **Parsing**: `unstructured[pdf]`, `pandas` (table structure), `pdf2image`
- **OCR**: `pytesseract` / `paddleocr`
- **ML / NLP**: `sentence-transformers` (MiniLM + CLIP), `transformers`
  (BLIP captioning), `cross-encoder` reranking, `rank-bm25`
- **Vector Store**: ChromaDB
- **LLM**: Groq (primary) / Hugging Face Inference API (fallback)
- **Infrastructure**: Docker
- **Testing**: `pytest` (offline suite covering core logic)

## 📁 Project Structure

```text
multimodal_rag/
├── app.py                 # Streamlit web dashboard
├── config.py              # all tunables (models, paths, thresholds)
├── Dockerfile
├── requirements.txt
├── sample_documents/       # zero-setup demo PDF
├── scripts/
│   └── generate_sample_pdf.py
├── src/
│   ├── document_parser.py # PDF → text / table / image chunks
│   ├── ocr_engine.py      # scanned-page OCR fallback
│   ├── chart_analyzer.py  # chart/infographic → searchable description
│   ├── embeddings.py      # MiniLM (text) + CLIP (image) embedders
│   ├── vector_store.py    # ChromaDB wrapper
│   ├── retrieval.py       # hybrid search + reranking
│   ├── ingest_utils.py    # file hashing + ingestion manifest
│   ├── llm_client.py      # Groq + HF Inference generation
│   └── rag_pipeline.py    # orchestration: ingest() / answer()
├── tests/                 # offline unit tests
└── data/                  # uploads, extracted images, chroma_db (gitignored)
```

## ⚙️ Local Development Setup

To run this project locally:

1. **Install system dependencies** (Debian/Ubuntu)
   ```bash
   sudo apt-get install -y poppler-utils tesseract-ocr libgl1
   ```
2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment**
   ```bash
   cp .env.example .env
   # Add your preferred LLM API key (e.g. GROQ_API_KEY)
   ```
4. **Run the Streamlit dashboard**
   ```bash
   streamlit run app.py
   ```
   *Dashboard will be available at http://localhost:8501*

A sample report auto-loads on first run — ask "What was North America's Q3
revenue?" straight away, or upload your own PDF.

## 🐳 Docker Deployment

To spin up the application in an isolated container:

1. **Build the image**
   ```bash
   docker build -t reportiq .
   ```
2. **Run the container**
   ```bash
   docker run -d -p 8501:8501 --env-file .env -v $(pwd)/data:/app/data reportiq
   ```
   *The app will be accessible at http://localhost:8501*

## 🧪 Testing

The project includes an offline test suite covering parsing, hybrid
search fusion, and ingestion dedup logic — no network calls, no PDF
parsing, no model downloads required.

```bash
pytest tests/ -v
```

17 tests, runs in under a second.

## Notes & Limitations

- No open model reads exact numeric values off a chart reliably —
  `chart_analyzer.py` combines captioning + OCR + a chart-type heuristic
  for retrieval and approximate description, not pixel-precise extraction.
  The system prompt tells the LLM to flag this rather than guess.
- First run downloads several model weights (BLIP, MiniLM, CLIP,
  reranker) — expect a delay.
- Single-process demo app (embedded Chroma, in-memory chat history) — for
  multi-user production use, run Chroma in server mode.

