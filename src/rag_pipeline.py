from sentence_transformers import SentenceTransformer
from vector_store import get_chroma_client, COLLECTION_NAME
import ollama
from document_filter import detect_document_filter
from reranker import blended_rerank
from query_rewriter import rewrite_query
from summarizer import compare_documents

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2:3b"

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

COMPARISON_KEYWORDS = ["compare", "comparison", "difference between", "differences between", "similarities", "versus", " vs "]


def retrieve_chunks(query, top_k=5, retrieve_pool_size=15):
    """Embeds the query, retrieves a wider candidate pool from Chroma,
    optionally restricted to specific document(s), then blends embedding
    rank + reranker rank (RRF) down to top_k."""
    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embed_model.encode([query]).tolist()

    matched_docs = detect_document_filter(query)

    where_clause = None
    if len(matched_docs) == 1:
        where_clause = {"document_name": matched_docs[0]}
    elif len(matched_docs) > 1:
        where_clause = {"document_name": {"$in": matched_docs}}

    if where_clause:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=retrieve_pool_size,
            where=where_clause
        )
    else:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=retrieve_pool_size
        )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "document_name": meta["document_name"],
            "page_number": meta["page_number"]
        })

    chunks = blended_rerank(query, chunks, top_k=top_k)
    return chunks


def build_prompt(query, chunks):
    """Builds a grounded prompt using retrieved chunks as context."""
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[Source: {c['document_name']}, Page {c['page_number']}]\n{c['text']}")

    context = "\n\n".join(context_blocks)

    prompt = f"""Answer the question using ONLY the context provided below.
Give a single, clear, complete sentence or short paragraph as your answer.
If the context fully or partially answers the question, answer confidently using what is available — do not add caveats about missing details unless truly nothing relevant is present.
If the answer is genuinely not present in the context at all, say only: "I could not find this information in the uploaded documents." Do not mix a correct answer with this phrase in the same response.

Context:
{context}

Question: {query}

Answer:"""
    return prompt


def ask_rag(query, chat_history=None, top_k=5):
    """
    Standard RAG pipeline with conversation memory:
    rewrite (if needed) -> retrieve -> rerank -> build prompt -> LLM answer.
    """
    if chat_history is None:
        chat_history = []

    standalone_query = rewrite_query(query, chat_history)

    chunks = retrieve_chunks(standalone_query, top_k=top_k)
    prompt = build_prompt(standalone_query, chunks)

    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )

    answer = response["message"]["content"]

    seen = set()
    unique_sources = []
    for c in chunks:
        key = (c["document_name"], c["page_number"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(c)

    return answer, unique_sources, standalone_query


def is_comparison_query(query):
    """Simple keyword-based check for comparison intent."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in COMPARISON_KEYWORDS)


def handle_query(query, chat_history=None, top_k=5):
    """
    Top-level entry point: routes to comparison handling if the query
    is a comparison question referencing 2+ specific documents,
    otherwise falls back to the standard RAG pipeline.
    """
    if chat_history is None:
        chat_history = []

    matched_docs = detect_document_filter(query)

    if is_comparison_query(query) and len(matched_docs) >= 2:
        print(f"[Routing] Detected comparison query across: {matched_docs}")
        comparison_answer = compare_documents(matched_docs[0], matched_docs[1])
        sources = [{"document_name": d, "page_number": "whole document"} for d in matched_docs[:2]]
        return comparison_answer, sources, query

    return ask_rag(query, chat_history=chat_history, top_k=top_k)


if __name__ == "__main__":
    query = "What text-to-SQL model architecture does RESDSQL use?"
    print(f"Question: {query}\n")

    answer, sources, _ = handle_query(query)

    print(f"\nAnswer:\n{answer}\n")
    print("--- Sources used ---")
    for c in sources:
        print(f"- {c['document_name']}, {c['page_number']}")