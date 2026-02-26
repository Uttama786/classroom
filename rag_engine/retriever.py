"""
RAG Retriever — loads the FAISS index and retrieves relevant chunks.
"""

import pickle
import pathlib
from typing import List, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "rag_engine" / "saved_index"

_index = None
_chunks = None
_model = None


def _load_resources():
    """Lazy-load the FAISS index, chunks, and embedding model."""
    global _index, _chunks, _model
    if _index is not None:
        return True

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"[Retriever] Missing dependency: {e}")
        return False

    index_path = INDEX_DIR / "index.faiss"
    chunks_path = INDEX_DIR / "chunks.pkl"

    if not index_path.exists() or not chunks_path.exists():
        print("[Retriever] Index not found. Run: python manage.py build_rag_index")
        return False

    _index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        _chunks = pickle.load(f)
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"[Retriever] Index loaded — {_index.ntotal} vectors")
    return True


def get_context(
    query: str,
    top_k: int = 5,
    subject_filter: Optional[str] = None,
) -> List[dict]:
    """
    Retrieve top_k most relevant chunks for the query.
    Optionally filter by subject code (e.g., 'DS', 'PY').
    Returns list of dicts: {text, source, subject, score}
    """
    if not _load_resources():
        return []

    import numpy as np

    query_embedding = _model.encode([query], convert_to_numpy=True).astype("float32")
    import faiss as faiss_module
    faiss_module.normalize_L2(query_embedding)

    # Search — retrieve extra candidates if filtering
    search_k = top_k * 4 if subject_filter else top_k
    scores, indices = _index.search(query_embedding, min(search_k, _index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = _chunks[idx].copy()
        chunk["score"] = float(score)

        # Apply subject filter
        if subject_filter:
            # Match by code prefix or full code
            chunk_subj = chunk.get("subject", "").upper()
            filter_code = subject_filter.upper()
            if chunk_subj and not chunk_subj.startswith(filter_code):
                continue

        results.append(chunk)
        if len(results) >= top_k:
            break

    return results
