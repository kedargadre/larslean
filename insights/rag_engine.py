"""
Hybrid RAG engine for the Léarslán AI Advisor.

TF-IDF keyword retrieval — fast, no external API required.
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 100
_FINAL_TOP_K = 3
_MIN_SIMILARITY = 0.05

_DOC_DIR = Path(__file__).parent.parent / "data" / "documents"
_METADATA_PATH = _DOC_DIR / "doc_metadata.json"


class HybridRAGEngine:
    """TF-IDF retrieval engine."""

    def __init__(self, doc_dir: Path = _DOC_DIR, metadata_path: Path = _METADATA_PATH):
        self.doc_dir = doc_dir
        self.metadata: dict = {}
        self.chunks: list[dict] = []
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None

        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self._load_and_chunk()
        self._build_tfidf_index()

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _load_and_chunk(self):
        """Sliding-window chunking with source metadata."""
        for fp in sorted(self.doc_dir.glob("*.*")):
            if fp.suffix not in (".txt", ".md"):
                continue
            text = fp.read_text(encoding="utf-8")
            meta = self.metadata.get(fp.name, {})

            for i in range(0, len(text), _CHUNK_SIZE - _CHUNK_OVERLAP):
                chunk_text = text[i : i + _CHUNK_SIZE].strip()
                if len(chunk_text) < 50:
                    continue
                self.chunks.append({
                    "content": chunk_text,
                    "source_file": fp.name,
                    "source_title": meta.get("title", fp.stem.replace("_", " ").title()),
                    "source_url": meta.get("source_url", ""),
                    "authority": meta.get("authority", ""),
                    "chunk_index": len(self.chunks),
                })

        logger.info("RAG: loaded %d chunks from %s", len(self.chunks), self.doc_dir)

    def _build_tfidf_index(self):
        if not self.chunks:
            return
        docs = [c["content"] for c in self.chunks]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(docs)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = _FINAL_TOP_K) -> list[dict]:
        """TF-IDF retrieval — returns top_k most relevant chunks."""
        if not self.chunks or self.tfidf_matrix is None:
            return []

        query_vec = self.tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        candidate_indices = np.argsort(tfidf_scores)[-top_k:][::-1]
        candidate_indices = [i for i in candidate_indices if tfidf_scores[i] > _MIN_SIMILARITY]

        results = []
        for idx in candidate_indices:
            chunk = self.chunks[idx]
            results.append({
                "content": chunk["content"],
                "source_file": chunk["source_file"],
                "source_title": chunk["source_title"],
                "source_url": chunk["source_url"],
                "authority": chunk["authority"],
                "score": round(float(tfidf_scores[idx]), 4),
                "method": "tfidf",
            })
        return results


# ── Module-level singleton ────────────────────────────────────────────────────

_engine: HybridRAGEngine | None = None


def get_rag_engine() -> HybridRAGEngine:
    """Lazy singleton — built once, reused across Streamlit reruns."""
    global _engine
    if _engine is None:
        _engine = HybridRAGEngine()
    return _engine


def reset_rag_engine():
    """Force rebuild."""
    global _engine
    _engine = None
