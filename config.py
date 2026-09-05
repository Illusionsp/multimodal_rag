"""
Central configuration for the Multimodal RAG system.
All tunables live here so the rest of the codebase never hardcodes paths/models.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
IMAGE_DIR = DATA_DIR / "images"          # extracted table/chart/figure crops
CHROMA_DIR = DATA_DIR / "chroma_db"      # persistent vector store

for d in (DATA_DIR, UPLOAD_DIR, IMAGE_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# "hi_res" gives the best table/figure detection (uses a layout model) but is
# heavier and needs extra system deps (poppler, detectron2/yolox weights).
# "fast" only does text extraction (no table structure). "ocr_only" forces
# OCR on every page - use for scanned documents.
UNSTRUCTURED_STRATEGY = os.getenv("UNSTRUCTURED_STRATEGY", "hi_res")
INFER_TABLE_STRUCTURE = True
EXTRACT_IMAGE_BLOCK_TYPES = ["Image", "Table"]

# Page is treated as "scanned" (falls back to full-page OCR) when unstructured
# extracts fewer than this many characters of text from it.
SCANNED_PAGE_CHAR_THRESHOLD = 20

# Chunking
CHUNK_SIZE = 800          # characters, for narrative text
CHUNK_OVERLAP = 120

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
# "paddleocr" (better accuracy, heavier install) or "tesseract" (lighter,
# needs the `tesseract-ocr` system package).
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract")
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "en")  # paddleocr lang code, e.g. "en"
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng")

# ---------------------------------------------------------------------------
# Vision / chart understanding
# ---------------------------------------------------------------------------
# Small, free, open-weight captioning model used to turn a chart/figure image
# into a text description that can be embedded and searched like any other
# text chunk. Swap for a larger BLIP-2 / LLaVA checkpoint if you have the GPU
# budget for it.
CAPTION_MODEL_NAME = os.getenv("CAPTION_MODEL_NAME", "Salesforce/blip-image-captioning-base")
CHART_CAPTION_MAX_NEW_TOKENS = 60

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# CLIP model used for pure visual similarity search (optional secondary index)
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "sentence-transformers/clip-ViT-B-32")
EMBED_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
TEXT_COLLECTION = "documents"   # unified text/table/chart-description space (MiniLM)
IMAGE_COLLECTION = "images"     # pure visual similarity space (CLIP)
DEFAULT_TOP_K = 5

# ---------------------------------------------------------------------------
# Retrieval quality: hybrid search + reranking
# ---------------------------------------------------------------------------
# Pure dense (embedding) search misses exact tokens that matter a lot in
# tables/charts - specific numbers, codes, labels. Hybrid search fuses dense
# similarity with BM25 keyword matching before reranking.
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# How many candidates to pull from each retriever before fusion/reranking,
# relative to the final top_k the user/UI asked for.
RETRIEVAL_CANDIDATE_MULTIPLIER = 4
RRF_K = 60  # reciprocal rank fusion constant (standard default)

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
# Manifest of already-ingested file hashes, so re-uploading the same PDF (or
# restarting the app) doesn't reprocess it from scratch.
INGEST_MANIFEST_PATH = DATA_DIR / "ingest_manifest.json"

# ---------------------------------------------------------------------------
# LLM (generation)
# ---------------------------------------------------------------------------
# Free-tier Groq API. Get a key at https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Fallback: free Hugging Face Inference API if no Groq key is set.
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

GENERATION_TEMPERATURE = 0.2
GENERATION_MAX_TOKENS = 1024
