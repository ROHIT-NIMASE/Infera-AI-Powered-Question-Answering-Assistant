from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker_model = CrossEncoder(RERANKER_MODEL_NAME)


def rerank_chunks(query, chunks, top_k=5):
    """
    Re-scores retrieved chunks using a cross-encoder for more accurate
    query-chunk relevance, and returns the top_k best chunks.
    """
    if not chunks:
        return chunks

    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker_model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


def blended_rerank(query, chunks, top_k=5, k=60):
    """
    Combines embedding-based rank and cross-encoder rerank using
    Reciprocal Rank Fusion (RRF), rather than trusting either signal alone.
    """
    if not chunks:
        return chunks

    embedding_ranks = {id(c): rank for rank, c in enumerate(chunks, start=1)}

    pairs = [(query, c["text"]) for c in chunks]
    rerank_scores = reranker_model.predict(pairs)

    scored_chunks = list(zip(chunks, rerank_scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    reranker_ranks = {id(c): rank for rank, (c, _) in enumerate(scored_chunks, start=1)}

    for c in chunks:
        emb_rank = embedding_ranks[id(c)]
        rerank_rank = reranker_ranks[id(c)]
        c["rrf_score"] = (1 / (k + emb_rank)) + (1 / (k + rerank_rank))
        c["embedding_rank"] = emb_rank
        c["reranker_rank"] = rerank_rank

    blended = sorted(chunks, key=lambda c: c["rrf_score"], reverse=True)
    return blended[:top_k]


if __name__ == "__main__":
    query = "How was the BIRD dataset constructed?"
    test_chunks = [
        {"text": "Table 2: Execution Accuracy of models on BIRD.", "document_name": "test", "page_number": 1},
        {"text": "We decouple the question and SQL annotation procedures to make the situation more realistic.", "document_name": "test", "page_number": 2},
        {"text": "Who funded the creation of the dataset? Alibaba DAMO Academy.", "document_name": "test", "page_number": 3},
    ]

    results = rerank_chunks(query, test_chunks, top_k=3)
    for r in results:
        print(f"Score: {r['rerank_score']:.4f} | {r['text'][:80]}")