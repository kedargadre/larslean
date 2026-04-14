"""
Hybrid RAG engine for the Léarslán AI Advisor.

Two-stage retrieval:
  Stage 1 — TF-IDF keyword search (fast, free, no API call) → top-K candidates
  Stage 2 — Gemini text-embedding-004 reranking (semantic) → top-N final chunks

Falls back to TF-IDF-only if Gemini embeddings are unavailable.
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
_TFIDF_CANDIDATES = 10
_FINAL_TOP_K = 3
_MIN_SIMILARITY = 0.05

_DOC_DIR = Path(__file__).parent.parent / "data" / "documents"
_METADATA_PATH = _DOC_DIR / "doc_metadata.json"


class HybridRAGEngine:
    """Two-stage retrieval: TF-IDF candidate selection + Gemini embedding rerank."""

    def __init__(self, doc_dir: Path = _DOC_DIR, metadata_path: Path = _METADATA_PATH):
        self.doc_dir = doc_dir
        self.metadata: dict = {}
        self.chunks: list[dict] = []
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.chunk_embeddings: np.ndarray | None = None
        self._gemini_available = False

        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self._load_and_chunk()
        self._build_tfidf_index()
        self._build_gemini_embeddings()

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

    def _build_gemini_embeddings(self):
        """Embed all chunks with Gemini embedding model at startup. Single batch call."""
        if not self.chunks:
            return
        try:
            from google import genai
            from config import GEMINI_API_KEY
            if not GEMINI_API_KEY:
                raise ValueError("No GEMINI_API_KEY")

            client = genai.Client(api_key=GEMINI_API_KEY)
            texts = [c["content"] for c in self.chunks]

            # Single batch call — all chunks at once
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
            )
            self.chunk_embeddings = np.array([emb.values for emb in result.embeddings])
            self._gemini_available = True
            logger.info("RAG: Gemini embeddings built for %d chunks (1 API call)", len(texts))
        except Exception as e:
            logger.warning("RAG: Gemini embeddings unavailable, TF-IDF only. %s", e)
            self._gemini_available = False

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = _FINAL_TOP_K) -> list[dict]:
        """
        Hybrid retrieval: TF-IDF first pass → Gemini rerank.

        Returns list of dicts with keys:
            content, source_file, source_title, source_url, authority, score
        """
        if not self.chunks or self.tfidf_matrix is None:
            return []

        # Stage 1: TF-IDF candidate selection
        query_vec = self.tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        candidate_count = min(_TFIDF_CANDIDATES, len(self.chunks))
        candidate_indices = np.argsort(tfidf_scores)[-candidate_count:][::-1]
        candidate_indices = [i for i in candidate_indices if tfidf_scores[i] > _MIN_SIMILARITY]

        if not candidate_indices:
            # No TF-IDF matches — return empty
            return []

        # Stage 2: Gemini semantic rerank (if available)
        if self._gemini_available and self.chunk_embeddings is not None:
            try:
                reranked = self._gemini_rerank(query, candidate_indices)
                return reranked[:top_k]
            except Exception as e:
                logger.warning("RAG: Gemini rerank failed, using TF-IDF order. %s", e)

        # Fallback: TF-IDF order
        results = []
        for idx in candidate_indices[:top_k]:
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

    def _gemini_rerank(self, query: str, candidate_indices: list[int]) -> list[dict]:
        """Rerank TF-IDF candidates using Gemini embedding cosine similarity."""
        from google import genai
        from config import GEMINI_API_KEY

        client = genai.Client(api_key=GEMINI_API_KEY)
        q_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
        )
        q_embedding = np.array(q_result.embeddings[0].values).reshape(1, -1)

        # Cosine similarity against candidate chunk embeddings
        candidate_embeddings = self.chunk_embeddings[candidate_indices]
        similarities = cosine_similarity(q_embedding, candidate_embeddings).flatten()

        # Sort by semantic similarity descending
        ranked_order = np.argsort(similarities)[::-1]

        results = []
        for rank_pos in ranked_order:
            idx = candidate_indices[rank_pos]
            chunk = self.chunks[idx]
            results.append({
                "content": chunk["content"],
                "source_file": chunk["source_file"],
                "source_title": chunk["source_title"],
                "source_url": chunk["source_url"],
                "authority": chunk["authority"],
                "score": round(float(similarities[rank_pos]), 4),
                "method": "hybrid",
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
    """Force rebuild — call when API key changes."""
    global _engine
    _engine = None
