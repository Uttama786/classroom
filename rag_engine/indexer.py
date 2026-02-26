"""
RAG Indexer — builds FAISS index from:
  1. Subject knowledge txt files (rag_knowledge/)
  2. Uploaded study materials (PDFs in media/)
  3. Quiz questions from the database
"""

import os
import sys
import pickle
import pathlib
from typing import List, Tuple

# Add project root to path so Django ORM works
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHUNK_SIZE = 400      # characters per chunk
CHUNK_OVERLAP = 60    # overlap between consecutive chunks


def chunk_text(text: str, source: str, subject_code: str = "") -> List[dict]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({
                "text": chunk,
                "source": source,
                "subject": subject_code,
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_knowledge_files(knowledge_dir: pathlib.Path) -> List[dict]:
    """Load subject .txt files."""
    all_chunks = []
    if not knowledge_dir.exists():
        print(f"[Indexer] Knowledge dir not found: {knowledge_dir}")
        return all_chunks

    for txt_file in knowledge_dir.glob("*.txt"):
        subject_code = txt_file.stem  # DS, PY, WD, CN, DSC, AIML
        text = txt_file.read_text(encoding="utf-8")
        chunks = chunk_text(text, source=f"Subject Notes: {subject_code}", subject_code=subject_code)
        all_chunks.extend(chunks)
        print(f"[Indexer] Loaded {len(chunks)} chunks from {txt_file.name}")

    return all_chunks


def load_pdf_materials(media_dir: pathlib.Path) -> List[dict]:
    """Extract text from uploaded PDF study materials."""
    all_chunks = []
    try:
        import pdfplumber
    except ImportError:
        print("[Indexer] pdfplumber not installed, skipping PDF indexing.")
        return all_chunks

    pdf_paths = list(media_dir.rglob("*.pdf"))
    print(f"[Indexer] Found {len(pdf_paths)} PDF(s) to index...")

    for pdf_path in pdf_paths:
        try:
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            full_text = "\n".join(text_parts)
            if full_text.strip():
                chunks = chunk_text(full_text, source=f"Study Material: {pdf_path.name}")
                all_chunks.extend(chunks)
                print(f"[Indexer]   Indexed {len(chunks)} chunks from {pdf_path.name}")
        except Exception as e:
            print(f"[Indexer]   Failed to read {pdf_path.name}: {e}")

    return all_chunks


def load_quiz_questions() -> List[dict]:
    """Load quiz questions and answers from the database."""
    all_chunks = []
    try:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flipped_classroom_project.settings')
        import django
        django.setup()
        from flipped_app.models import QuizQuestion

        qs = QuizQuestion.objects.select_related('quiz', 'quiz__subject').all()
        for q in qs:
            subject_code = q.quiz.subject.code if q.quiz.subject else ""
            text = (
                f"Question: {q.question_text}\n"
                f"Options: A) {q.option_a}  B) {q.option_b}  "
                f"C) {q.option_c}  D) {q.option_d}\n"
                f"Correct Answer: {q.correct_answer}\n"
                f"Explanation: {getattr(q, 'explanation', 'N/A')}"
            )
            all_chunks.append({
                "text": text,
                "source": f"Quiz: {q.quiz.title}",
                "subject": subject_code,
            })
        print(f"[Indexer] Loaded {len(all_chunks)} quiz questions from DB")
    except Exception as e:
        print(f"[Indexer] Could not load quiz questions: {e}")

    return all_chunks


def build_index():
    """Build FAISS index and save to disk."""
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
    except ImportError as e:
        print(f"[Indexer] Missing dependency: {e}")
        print("[Indexer] Run: pip install sentence-transformers faiss-cpu")
        return

    knowledge_dir = PROJECT_ROOT / "rag_knowledge"
    media_dir = PROJECT_ROOT / "media"
    save_dir = PROJECT_ROOT / "rag_engine" / "saved_index"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect all chunks
    all_chunks = []
    all_chunks.extend(load_knowledge_files(knowledge_dir))
    all_chunks.extend(load_pdf_materials(media_dir))
    all_chunks.extend(load_quiz_questions())

    if not all_chunks:
        print("[Indexer] No content to index!")
        return

    print(f"\n[Indexer] Total chunks: {len(all_chunks)}")
    print("[Indexer] Loading embedding model (all-MiniLM-L6-v2)...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in all_chunks]

    print("[Indexer] Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(save_dir / "index.faiss"))
    with open(save_dir / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\n[Indexer] ✅ Index saved to {save_dir}")
    print(f"[Indexer] Total vectors: {index.ntotal}")


if __name__ == "__main__":
    build_index()
