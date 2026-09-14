from pdf_loader import load_all_pdfs
from chunker_langchain import chunk_text_langchain as chunk_text

def build_chunks_from_pdfs():
    """
    Loads all PDFs, chunks each page's text, and attaches
    document_name + page_number metadata to every chunk.
    Returns a list of dicts:
    {"document_name": ..., "page_number": ..., "chunk_text": ..., "chunk_id": ...}
    """
    pages = load_all_pdfs()
    all_chunks = []
    chunk_counter = 0

    for page in pages:
        page_chunks = chunk_text(page["text"])

        for chunk in page_chunks:
            chunk_counter += 1
            all_chunks.append({
                "chunk_id": chunk_counter,
                "document_name": page["document_name"],
                "page_number": page["page_number"],
                "chunk_text": chunk
            })

    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks_from_pdfs()

    print(f"Total chunks created: {len(chunks)}\n")
    print("--- Sample chunk ---")
    print(chunks[0])
    print(f"\nChunk length (chars): {len(chunks[0]['chunk_text'])}")