import chromadb
from embedder import build_embedded_chunks
from embedder import build_embedded_chunks_from_files

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "document_chunks"


def get_chroma_client():
    """Creates a persistent Chroma client that saves data to disk at DB_PATH."""
    return chromadb.PersistentClient(path=DB_PATH)


def add_documents_to_store(uploaded_file_paths):
    """
    Processes new PDF files and adds their embedded chunks to the existing
    Chroma collection, WITHOUT deleting what's already there.
    """
    client = get_chroma_client()
    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing:
        collection = client.get_collection(COLLECTION_NAME)
    else:
        collection = client.create_collection(name=COLLECTION_NAME)

    embedded_chunks = build_embedded_chunks_from_files(uploaded_file_paths)

    if not embedded_chunks:
        return 0

    # Use a running max chunk_id offset so new IDs never collide with existing ones
    existing_count = collection.count()

    ids = [str(existing_count + i + 1) for i in range(len(embedded_chunks))]
    embeddings = [c["embedding"].tolist() for c in embedded_chunks]
    documents = [c["chunk_text"] for c in embedded_chunks]
    metadatas = [
        {"document_name": c["document_name"], "page_number": c["page_number"]}
        for c in embedded_chunks
    ]

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(embedded_chunks)


def clear_all_documents():
    """Deletes the entire collection - used by the 'Clear all' button."""
    client = get_chroma_client()
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

def build_vector_store():
    """
    Embeds all PDF chunks and stores them in a Chroma collection,
    including their text and metadata (document name, page number).
    """
    client = get_chroma_client()

    # If the collection already exists from a previous run, delete it first
    # so we don't end up with duplicate entries when re-running this script.
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' to rebuild fresh.")

    collection = client.create_collection(name=COLLECTION_NAME)

    embedded_chunks = build_embedded_chunks()

    ids = [str(c["chunk_id"]) for c in embedded_chunks]
    embeddings = [c["embedding"].tolist() for c in embedded_chunks]  # Chroma needs plain lists, not numpy arrays
    documents = [c["chunk_text"] for c in embedded_chunks]
    metadatas = [
        {"document_name": c["document_name"], "page_number": c["page_number"]}
        for c in embedded_chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Stored {collection.count()} chunks in Chroma collection '{COLLECTION_NAME}'.")
    return collection


if __name__ == "__main__":
    collection = build_vector_store()
    print(f"\nCollection ready. Total items: {collection.count()}")