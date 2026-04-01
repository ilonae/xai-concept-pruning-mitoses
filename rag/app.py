"""
Research Explanatory Assistant using Streamlit app

Layout:
  Left panel: query input field, LLM answer below - and explanation cards
  Right panel: UMAP embedding space
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer

import pipeline as rag
from explain import (
    build_umap_projection,
    heatmap_html,
    keyword_overlap,
    project_query,
    sentence_relevance,
)

st.set_page_config(
    page_title="Research Explanatory Assistant using Streamlit",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Research Explanatory Assistant")
st.caption(
    "Explainable RAG over research papers"
    "multilingual (EN + DE) - and fully local")

@st.cache_resource(show_spinner="Loading indexing and models…")
def load_all():
    embed_model_name = rag.EMBED_MODEL_NAME

    # Groq API key: Streamlit secrets (cloud) - env var (local .env) - None (Ollama)
    groq_api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None

    # Configure LlamaIndex settings (Groq if key present, else Ollama)
    rag.configure_settings(groq_api_key=groq_api_key)
    rag.ensure_papers()

    # Build or load vector index
    index, chroma_collection = rag.build_or_load_index()

    # Sentence-transformer model (used for heatmap + UMAP)
    st_model = SentenceTransformer(embed_model_name)

    # Pull all embeddings with metadata from ChromaDB for UMAP, fetching in batches to avoid SQLite's variable limit (999)
    total = chroma_collection.count()
    batch_size = 100
    all_embeddings, all_documents, all_metadatas = [], [], []

    for offset in range(0, total, batch_size):
        batch = chroma_collection.get(
            include=["embeddings", "documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        all_embeddings.extend(batch["embeddings"])
        all_documents.extend(batch["documents"])
        all_metadatas.extend(batch["metadatas"])

    embeddings = np.array(all_embeddings) 
    documents = all_documents              # list of chunk texts
    metadatas = all_metadatas              # list of dicts

    # Fit UMAP on all chunk embeddings (slow once, then cached)
    with st.spinner("Fitting UMAP projection…"):
        reducer, coords_2d = build_umap_projection(embeddings)

    return index, st_model, reducer, coords_2d, documents, metadatas


try:
    index, st_model, umap_reducer, coords_2d, all_docs, all_meta = load_all()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=3, max_value=10, value=5)
    show_heatmap = st.toggle("Sentence heatmap", value=True)
    show_keywords = st.toggle("Keyword overlap", value=True)
    st.divider()
    st.caption(f"Index: {len(all_docs)} chunks")
    sources = sorted({m.get("file_name", "?") for m in all_meta if m})
    st.caption("Sources:\n" + "\n".join(f"· {s}" for s in sources))

left, right = st.columns([1, 1], gap="large")

with left:
    query = st.text_input(
        "Ask a question:",
        placeholder="How does CRP differ from standard LRP?",
    )

if not query:
    with right:
        # Show full UMAP without a query point when idle
        unique_sources = sorted({m.get("file_name", "?") for m in all_meta if m})
        fig = go.Figure()
        for src in unique_sources:
            mask = [m.get("file_name") == src for m in all_meta if m]
            x = coords_2d[[i for i, v in enumerate(mask) if v], 0]
            y = coords_2d[[i for i, v in enumerate(mask) if v], 1]
            texts = [all_docs[i][:80] + "…" for i, v in enumerate(mask) if v]
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(size=5, opacity=0.4),
                name=src, text=texts, hoverinfo="text+name",
            ))
        fig.update_layout(
            height=520,
            title="All chunk embeddings (UMAP: 2D)",
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Enter a query to see where it lands in the embedding space.")
    st.stop()

# Running the RAG query

with st.spinner("Retrieving and generating…"):
    response = rag.query(index, query, top_k=top_k)
    query_2d = project_query(query, st_model, umap_reducer)

source_nodes = response.source_nodes

# Left panel: answer and explanation cards

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

            # 3 — keyword overlap
            if show_keywords:
                kws = keyword_overlap(query, node.text)
                if kws:
                    pills = " &nbsp;".join(
                        f'<code style="background:#f0f0f0; padding:1px 6px; border-radius:4px;">{k}</code>'
                        for k in kws[:10]
                    )
                    st.markdown(
                        f'<p style="margin:0 0 6px 0; font-size:0.85em;">Shared concepts: {pills}</p>',
                        unsafe_allow_html=True,
                    )

            # 2 — sentence heatmap
            if show_heatmap:
                sent_scores = sentence_relevance(query, node.text, st_model)
                st.markdown(heatmap_html(sent_scores), unsafe_allow_html=True)
            else:
                st.write(node.text)

with right:
    st.subheader("Embedding Space")

    # Identify which ChromaDB entries were retrieved
    retrieved_node_ids = {node.node_id for node in source_nodes}
    all_ids = [
        m.get("_node_id") or m.get("id") or ""
        for m in all_meta if m
    ]
    # Fallback: match by first 80 chars of text if IDs aren't stored
    retrieved_texts_prefix = {node.text[:80] for node in source_nodes}
    is_retrieved = [
        doc[:80] in retrieved_texts_prefix
        for doc in all_docs
    ]

    unique_sources = sorted({m.get("file_name", "?") for m in all_meta if m})
    fig = go.Figure()

    # Non-retrieved chunks, colored by source
    for src in unique_sources:
        mask = [
            m.get("file_name") == src and not is_retrieved[i]
            for i, m in enumerate(all_meta)
            if m is not None
        ]
        idxs = [i for i, v in enumerate(mask) if v]
        if not idxs:
            continue
        fig.add_trace(go.Scatter(
            x=coords_2d[idxs, 0],
            y=coords_2d[idxs, 1],
            mode="markers",
            marker=dict(size=5, opacity=0.35),
            name=src,
            text=[all_docs[i][:80] + "…" for i in idxs],
            hoverinfo="text+name",
        ))

    # Retrieved chunks (highlighted)
    ret_idxs = [i for i, v in enumerate(is_retrieved) if v]
    if ret_idxs:
        fig.add_trace(go.Scatter(
            x=coords_2d[ret_idxs, 0],
            y=coords_2d[ret_idxs, 1],
            mode="markers",
            marker=dict(
                size=14,
                color="orange",
                line=dict(width=2, color="black"),
            ),
            name="Retrieved",
            text=[all_docs[i][:80] + "…" for i in ret_idxs],
            hoverinfo="text+name",
        ))

    # Query point
    fig.add_trace(go.Scatter(
        x=[query_2d[0]],
        y=[query_2d[1]],
        mode="markers+text",
        marker=dict(size=18, color="crimson", symbol="star"),
        text=["Query"],
        textposition="top center",
        name="Query",
        hovertext=f"Query: {query}",
        hoverinfo="text",
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
        "★ red = query &nbsp;·&nbsp; ● orange = retrieved chunks "
        "&nbsp;·&nbsp; · dots = all chunks by source paper"
    )
