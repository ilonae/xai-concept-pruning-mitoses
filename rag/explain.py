"""
Explanation logic for the RAG pipeline.

Three functions, one per concept:
  - sentence_relevance()   - 2: answers which sentences in a chunk are most relevant
  - keyword_overlap()      - 3: answers what terms connect query and chunk
  - build_umap_projection() - 1: projects all embeddings to 2D
"""

import re

import numpy as np
from sentence_transformers import SentenceTransformer


# 2 — Sentence-level relevance heatmap

def sentence_relevance(query: str, chunk_text: str, model: SentenceTransformer) -> list[tuple[str, float]]:
    """
    Split a chunk into sentences, score each one against the query - returns [(sentence, cosine_similarity), ...] sorted by original order
    """
    raw = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
    sentences = [s.strip() for s in raw if len(s.strip()) > 15]

    if not sentences:
        return [(chunk_text, 1.0)]

    query_emb = model.encode(query, normalize_embeddings=True)
    sent_embs = model.encode(sentences, normalize_embeddings=True, batch_size=32)
    scores = [float(np.dot(query_emb, e)) for e in sent_embs]

    return list(zip(sentences, scores))


def heatmap_html(sentence_scores: list[tuple[str, float]]) -> str:
    """
    Render sentences as an HTML string with color proportional to relevance score. White (low), up to orange (high)
    """
    scores = [s for _, s in sentence_scores]
    lo, hi = min(scores), max(scores)

    def to_color(score: float) -> str:
        t = (score - lo) / (hi - lo) if hi > lo else 0.5
        t = max(0.0, min(1.0, t))
        # white → amber: (255,255,255) → (255,180,0)
        g = int(255 - t * 75)
        b = int(255 - t * 255)
        alpha = 0.15 + t * 0.65
        return f"rgba(255,{g},{b},{alpha:.2f})"

    parts = [
        f'<span style="background:{to_color(score)}; padding:2px 3px; '
        f'border-radius:3px; line-height:1.9;">{sent} </span>'
        for sent, score in sentence_scores
    ]
    return "".join(parts)

# 3 — Keyword overlap (shared concepts)

# Basic bilingual stopword list (EN and DE)
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "to", "for",
    "on", "with", "at", "by", "from", "as", "into", "that", "this", "it",
    "its", "and", "or", "but", "not", "what", "how", "why", "when", "where",
    "which", "who", "their", "they", "these", "those", "also", "such",
    # German
    "der", "die", "das", "ein", "eine", "und", "oder", "ist", "sind",
    "war", "waren", "wird", "werden", "hat", "haben", "im", "in", "an",
    "auf", "mit", "von", "zu", "für", "des", "dem", "den", "als", "auch",
    "durch", "nach", "über", "bei", "aus", "dass", "sich", "nicht", "wie",
}


def keyword_overlap(query: str, chunk_text: str, min_len: int = 4) -> list[str]:
    """
    Find meaningful shared terms between query and chunk text and filters stopwords and short tokens - Returns sorted list
    """
    def tokenize(text: str) -> set[str]:
        return {
            w for w in re.findall(r"\b\w+\b", text.lower())
            if len(w) >= min_len and w not in _STOPWORDS
        }

    return sorted(tokenize(query) & tokenize(chunk_text))

# 1 — Embedding space (UMAP projection)

def build_umap_projection(embeddings: np.ndarray):
    """
    Fit a UMAP reducer on a matrix of embeddings and returns reducer, 2D coordinates
    Cached by caller (e.g. @st.cache_resource)
    """
    import umap  # lazy import — only needed when function is called

    n = len(embeddings)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, n - 1),
        min_dist=0.1,
        metric="cosine",
        init="random",   # avoids eigenvector solver failure on small/similar datasets
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)
    return reducer, coords


def project_query(query: str, model: SentenceTransformer, reducer) -> np.ndarray:
    """
    Embed a query and project it into the fitted UMAP, returns am array of 2D coordinates
    """
    emb = model.encode(query, normalize_embeddings=True)
    return reducer.transform([emb])[0]
