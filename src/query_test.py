from sentence_transformers import SentenceTransformer
from vector_store import get_chroma_client, COLLECTION_NAME

MODEL_NAME = "all-MiniLM-L6-v2"


def search(query, top_k=5):
    """
    Embeds a query and retrieves the top_k most similar chunks from Chroma.
    """
    model = SentenceTransformer(MODEL_NAME)
    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results


def print_results(results):
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        print(f"\n--- Result {i} (distance: {dist:.4f}) ---")
        print(f"Source: {meta['document_name']}, Page: {meta['page_number']}")
        print(f"Text: {doc[:300]}...")


if __name__ == "__main__":
    query = "What text-to-SQL model architecture does RESDSQL use?"
    print(f"Query: \"{query}\"\n")

    results = search(query, top_k=5)
    print_results(results)