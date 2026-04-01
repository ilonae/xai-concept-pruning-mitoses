import numpy as np
import pytest
from fixtures import EVAL_QUERIES

# Thresholds — adjust 

HIT_RATE_THRESHOLD = 0.80   # at least 80% of queries must hit in top-5
MRR_THRESHOLD = 0.60        # average rank must be ≥ 0.60
TOP_K = 5


def _is_hit(retrieved_texts: list[str], expected_keywords: list[str]) -> bool:
    """True if any retrieved chunk contains at least one expected keyword."""
    for text in retrieved_texts:
        text_lower = text.lower()
        if any(kw in text_lower for kw in expected_keywords):
            return True
    return False


def _reciprocal_rank(retrieved_texts: list[str], expected_keywords: list[str]) -> float:
    """Reciprocal rank of the first retrieved chunk that contains an expected keyword."""
    for rank, text in enumerate(retrieved_texts, start=1):
        text_lower = text.lower()
        if any(kw in text_lower for kw in expected_keywords):
            return 1.0 / rank
    return 0.0

# Per-query retrieval tests

@pytest.mark.parametrize("eval_item", EVAL_QUERIES, ids=[q["query"][:40] for q in EVAL_QUERIES])
def test_query_has_hit_in_top_k(eval_item, in_memory_index):
    """Each query must retrieve at least one relevant chunk in the top-k results."""
    retriever = in_memory_index.as_retriever(similarity_top_k=TOP_K)
    nodes = retriever.retrieve(eval_item["query"])
    retrieved_texts = [n.text for n in nodes]

    hit = _is_hit(retrieved_texts, eval_item["expected_keywords"])

    assert hit, (
        f"\nQuery   : {eval_item['query']}\n"
        f"Keywords: {eval_item['expected_keywords']}\n"
        f"Top-{TOP_K} retrieved:\n"
        + "\n".join(f"  [{i+1}] {t[:100]}..." for i, t in enumerate(retrieved_texts))
    )


@pytest.mark.parametrize("eval_item", EVAL_QUERIES, ids=[q["query"][:40] for q in EVAL_QUERIES])
def test_query_reciprocal_rank(eval_item, in_memory_index):
    """Relevant chunk should appear in top-3 (MRR ≥ 0.33) for each query."""
    retriever = in_memory_index.as_retriever(similarity_top_k=TOP_K)
    nodes = retriever.retrieve(eval_item["query"])
    retrieved_texts = [n.text for n in nodes]

    rr = _reciprocal_rank(retrieved_texts, eval_item["expected_keywords"])

    assert rr >= 0.33, (
        f"\nQuery   : {eval_item['query']}\n"
        f"RR      : {rr:.3f} (need ≥ 0.33, i.e. relevant chunk in top-3)\n"
        f"Keywords: {eval_item['expected_keywords']}\n"
        f"Top-{TOP_K} retrieved:\n"
        + "\n".join(f"  [{i+1}] {t[:100]}..." for i, t in enumerate(retrieved_texts))
    )

# Aggregate quality gate 

def test_aggregate_hit_rate_above_threshold(in_memory_index):
    """
    Aggregate hit rate across all eval queries must exceed HIT_RATE_THRESHOLD
    """
    retriever = in_memory_index.as_retriever(similarity_top_k=TOP_K)
    hits = []

    for item in EVAL_QUERIES:
        nodes = retriever.retrieve(item["query"])
        retrieved_texts = [n.text for n in nodes]
        hits.append(int(_is_hit(retrieved_texts, item["expected_keywords"])))

    hit_rate = np.mean(hits)

    assert hit_rate >= HIT_RATE_THRESHOLD, (
        f"Hit rate {hit_rate:.2%} is below threshold {HIT_RATE_THRESHOLD:.2%}.\n"
        f"Per-query results: {list(zip([q['query'][:30] for q in EVAL_QUERIES], hits))}"
    )


def test_aggregate_mrr_above_threshold(in_memory_index):
    """
    Mean Reciprocal Rank across all eval queries must exceed MRR_THRESHOLD
    """
    retriever = in_memory_index.as_retriever(similarity_top_k=TOP_K)
    rrs = []

    for item in EVAL_QUERIES:
        nodes = retriever.retrieve(item["query"])
        retrieved_texts = [n.text for n in nodes]
        rrs.append(_reciprocal_rank(retrieved_texts, item["expected_keywords"]))

    mrr = np.mean(rrs)

    assert mrr >= MRR_THRESHOLD, (
        f"MRR {mrr:.3f} is below threshold {MRR_THRESHOLD:.3f}.\n"
        f"Per-query RR: {list(zip([q['query'][:30] for q in EVAL_QUERIES], [f'{r:.2f}' for r in rrs]))}"
    )

# Source diversity check

def test_retrieval_covers_multiple_sources(in_memory_index):
    """
    For a broad query, retrieved chunks should come from more than one source document
    """
    retriever = in_memory_index.as_retriever(similarity_top_k=TOP_K)
    nodes = retriever.retrieve("explainable AI concept attribution relevance propagation")
    sources = {n.metadata.get("file_name") for n in nodes}
    assert len(sources) > 1, (
        f"Expected results from multiple sources, got only: {sources}"
    )


# Score sanity checks

def test_similarity_scores_are_positive(in_memory_index):
    """Cosine similarity scores for on-topic queries should be positive."""
    retriever = in_memory_index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve("What is Concept Relevance Propagation?")
    for node in nodes:
        assert node.score is not None
        assert node.score > 0, f"Expected positive similarity score, got {node.score}"


def test_relevant_query_scores_higher_than_irrelevant(in_memory_index):
    """
    A domain-relevant query should produce higher top-1 similarity than a completely off-topic query
    """
    retriever = in_memory_index.as_retriever(similarity_top_k=1)

    on_topic = retriever.retrieve("Concept Relevance Propagation attribution network")
    off_topic = retriever.retrieve("weather forecast rainfall temperature humidity")

    score_on = on_topic[0].score if on_topic else 0.0
    score_off = off_topic[0].score if off_topic else 0.0

    assert score_on > score_off, (
        f"On-topic score ({score_on:.3f}) should exceed off-topic score ({score_off:.3f})"
    )
