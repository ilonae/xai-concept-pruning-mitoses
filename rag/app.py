"""
Research Explanatory Assistant — Streamlit app

Layout:
  Left panel  → query input · LLM answer · explanation cards (heatmap + keywords)
  Right panel → UMAP embedding space

Memory budget (Streamlit Community Cloud ~1 GB):
  - Embedding model loaded ONCE via HuggingFaceEmbedding; reused for sentence scoring
  - UMAP capped at MAX_UMAP_POINTS so we never load the full corpus into RAM
  - ChromaDB index stays on disk; only retrieved chunks are materialised
"""

import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import pipeline as rag
from explain import (
    build_umap_projection,
    heatmap_html,
    keyword_overlap,
    project_query,
    sentence_relevance,
)

# Maximum number of chunk embeddings loaded into memory for the UMAP plot.
# The full index stays in ChromaDB on disk; only this many points are visualised.
MAX_UMAP_POINTS = 150

st.set_page_config(
    page_title="XAI Paper Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("XAI Paper Assistant")
st.caption("Explainable RAG over XAI research papers · multilingual (EN + DE)")


# ---------------------------------------------------------------------------
# Load everything once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading index and models…")
def load_all():
    # --- Resolve Groq key ---
    groq_api_key = None
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.warning(
            "No GROQ_API_KEY found. Add it in **Manage app → Settings → Secrets**. "
            "Falling back to Ollama — this will fail on the cloud.",
            icon="⚠️",
        )

    # --- Configure LlamaIndex (returns HuggingFaceEmbedding) ---
    hf_embed = rag.configure_settings(groq_api_key=groq_api_key)

    # Reuse the SentenceTransformer already loaded inside HuggingFaceEmbedding.
    # This avoids loading the 270 MB model a second time.
    st_model = hf_embed._model

    # --- Papers + index ---
    rag.ensure_papers()
    index, chroma_collection = rag.build_or_load_index()

    # --- Fetch a capped sample of embeddings for UMAP ---
    # We never need the full corpus in RAM — MAX_UMAP_POINTS is enough
    # for a representative visualisation.
    total = chroma_collection.count()
    fetch_n = min(total, MAX_UMAP_POINTS)

    sample = chroma_collection.get(
        include=["embeddings", "documents", "metadatas"],
        limit=fetch_n,
    )

    embeddings = np.array(sample["embeddings"])
    umap_docs = sample["documents"]
    umap_meta = sample["metadatas"]

    # --- Fit UMAP on the capped sample ---
    with st.spinner("Fitting UMAP projection…"):
        reducer, coords_2d = build_umap_projection(embeddings)

    return index, st_model, reducer, coords_2d, umap_docs, umap_meta


try:
    index, st_model, umap_reducer, coords_2d, umap_docs, umap_meta = load_all()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=3, max_value=10, value=5)
    show_heatmap = st.toggle("Sentence heatmap", value=True)
    show_keywords = st.toggle("Keyword overlap", value=True)
    st.divider()
    sources = sorted({m.get("file_name", "?") for m in umap_meta if m})
    st.caption("Sources:\n" + "\n".join(f"· {s}" for s in sources))
    st.caption(f"Showing {len(umap_docs)} of index chunks in UMAP")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

left, right = st.columns([1, 1], gap="large")

with left:
    query = st.text_input(
        "Ask a question:",
        placeholder="How does CRP differ from standard LRP?",
    )

# Idle state — show UMAP without a query point
if not query:
    with right:
        fig = go.Figure()
        unique_sources = sorted({m.get("file_name", "?") for m in umap_meta if m})
        for src in unique_sources:
            idxs = [i for i, m in enumerate(umap_meta) if m and m.get("file_name") == src]
            if not idxs:
                continue
            fig.add_trace(go.Scatter(
                x=coords_2d[idxs, 0],
                y=coords_2d[idxs, 1],
                mode="markers",
                marker=dict(size=5, opacity=0.4),
                name=src,
                text=[umap_docs[i][:80] + "…" for i in idxs],
                hoverinfo="text+name",
            ))
        fig.update_layout(
            height=520,
            title="Chunk embeddings (UMAP · 2D)",
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Enter a query to see where it lands in the embedding space.")
    st.stop()


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

with st.spinner("Retrieving and generating…"):
    response = rag.query(index, query, top_k=top_k)
    query_2d = project_query(query, st_model, umap_reducer)

source_nodes = response.source_nodes

# --- Left: answer + explanation cards ---
with left:
    st.subheader("Answer")
    st.write(response.response)

    st.divider()
    st.subheader(f"Retrieved chunks — top {len(source_nodes)}")

    for i, node in enumerate(source_nodes):
        score = node.score or 0.0
        fname = node.metadata.get("file_name", "unknown")
        page = node.metadata.get("page_label", "?")
        label = f"[{i+1}]  {fname}  ·  p.{page}  ·  similarity {score:.3f}"

        with st.expander(label, expanded=(i == 0)):
            # Keyword overlap
            if show_keywords:
                kws = keyword_overlap(query, node.text)
                if kws:
                    pills = " &nbsp;".join(
                        f'<code style="background:#f0f0f0;padding:1px 6px;border-radius:4px;">{k}</code>'
                        for k in kws[:10]
                    )
                    st.markdown(
                        f'<p style="margin:0 0 6px 0;font-size:0.85em;">Shared concepts: {pills}</p>',
                        unsafe_allow_html=True,
                    )

            # Sentence heatmap
            if show_heatmap:
                sent_scores = sentence_relevance(query, node.text, st_model)
                st.markdown(heatmap_html(sent_scores), unsafe_allow_html=True)
            else:
                st.write(node.text)

# --- Right: UMAP ---
with right:
    st.subheader("Embedding Space")

    retrieved_prefixes = {node.text[:80] for node in source_nodes}
    is_retrieved = [doc[:80] in retrieved_prefixes for doc in umap_docs]

    unique_sources = sorted({m.get("file_name", "?") for m in umap_meta if m})
    fig = go.Figure()

    for src in unique_sources:
        idxs = [
            i for i, m in enumerate(umap_meta)
            if m and m.get("file_name") == src and not is_retrieved[i]
        ]
        if not idxs:
            continue
        fig.add_trace(go.Scatter(
            x=coords_2d[idxs, 0],
            y=coords_2d[idxs, 1],
            mode="markers",
            marker=dict(size=5, opacity=0.35),
            name=src,
            text=[umap_docs[i][:80] + "…" for i in idxs],
            hoverinfo="text+name",
        ))

    ret_idxs = [i for i, v in enumerate(is_retrieved) if v]
    if ret_idxs:
        fig.add_trace(go.Scatter(
            x=coords_2d[ret_idxs, 0],
            y=coords_2d[ret_idxs, 1],
            mode="markers",
            marker=dict(size=14, color="orange", line=dict(width=2, color="black")),
            name="Retrieved",
            text=[umap_docs[i][:80] + "…" for i in ret_idxs],
            hoverinfo="text+name",
        ))

    fig.add_trace(go.Scatter(
        x=[query_2d[0]], y=[query_2d[1]],
        mode="markers+text",
        marker=dict(size=18, color="crimson", symbol="star"),
        text=["Query"], textposition="top center",
        name="Query",
        hovertext=f"Query: {query}", hoverinfo="text",
    ))

    fig.update_layout(
        height=520,
        xaxis=dict(showticklabels=False, title=""),
        yaxis=dict(showticklabels=False, title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "★ red = query &nbsp;·&nbsp; ● orange = retrieved "
        "&nbsp;·&nbsp; · dots = sampled chunks by source"
    )
