from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DIR = PROJECT_ROOT / "data"
ARTICLES_DIR = DATA_DIR / "articles"
ENTITIES_DIR = DATA_DIR / "entities"
GRAPH_DIR = DATA_DIR / "graph"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Law versions
LAWS = {
    "2024": {
        "id": "2024",
        "name": "Luật Đất đai 2024",
        "short": "LĐĐ 2024",
        "number": "31/2024/QH15",
        "source_text": DOCS_DIR / "31-2024-qh15_1.txt",
    },
    "2013": {
        "id": "2013",
        "name": "Luật Đất đai 2013",
        "short": "LĐĐ 2013",
        "number": "45/2013/QH13",
        "source_text": DOCS_DIR / "VanBanGoc_45.2013.QH13.txt",
    },
}

# API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Embedding
EMBEDDING_MODEL = "dangvantuan/vietnamese-embedding"
EMBEDDING_DIM = 768

# Retrieval
TOP_K_VECTOR = 15
GRAPH_HOP_DEPTH = 2
TOP_K_FINAL = 8

# Ensure directories exist
for d in [ARTICLES_DIR, ENTITIES_DIR, GRAPH_DIR, EMBEDDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
