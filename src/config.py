import os

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# Paths
DATA_DIR = "data"
DB_PATH = "chroma_db"
COLLECTION = "docs"

# Models
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking / retrieval defaults (tune these via evaluate.py)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
USE_HYBRID = True      # BM25 + vector search fused with reciprocal rank fusion
USE_RERANKER = False   # cross-encoder reranking (slower, usually more accurate)

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")   # "anthropic", "ollama", or "none" (no LLM: shows matching passages only)
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

ABSTAIN_PHRASE = "I couldn't find that in the documents."
