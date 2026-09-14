from sentence_transformers import SentenceTransformer
from pdf_loader import load_all_pdfs, extract_text_from_pdf
import os

MODEL_NAME = "all-MiniLM-L6-v2"


def extract_title_from_page_text(page1_text, max_len=120):
    """
    Extracts a likely title from a document's first page text.
    Heuristic: skip boilerplate/copyright lines, take the first
    substantial line as the title.
    """
    lines = [l.strip() for l in page1_text.split("\n") if l.strip()]
    for line in lines:
        # Skip obvious boilerplate (copyright/permission notices, very short lines)
        lower = line.lower()
        if any(kw in lower for kw in ["permission", "copyright", "reproduce", "arxiv"]):
            continue
        if len(line) < 8:
            continue
        return line[:max_len]
    return lines[0][:max_len] if lines else "Untitled document"


def _build_title_map(pages):
    """Builds {document_name: title} using each document's first page_number==1 text."""
    title_map = {}
    for page in pages:
        if page["page_number"] == 1 and page["document_name"] not in title_map:
            title_map[page["document_name"]] = extract_title_from_page_text(page["text"])
    return title_map


def _chunk_and_embed(pages, chunk_text_fn, model):
    """
    Shared logic: chunks pages, prepends each chunk's document title for
    embedding purposes only (original chunk_text stays unchanged for storage/display).
    """
    title_map = _build_title_map(pages)

    all_chunks = []
    chunk_counter = 0

    for page in pages:
        page_chunks = chunk_text_fn(page["text"])
        for chunk in page_chunks:
            chunk_counter += 1
            all_chunks.append({
                "chunk_id": chunk_counter,
                "document_name": page["document_name"],
                "page_number": page["page_number"],
                "chunk_text": chunk  # original, unmodified - used for display/context
            })

    if not all_chunks:
        return []

    # Build enriched text ONLY for embedding computation - title + chunk content
    texts_for_embedding = [
        f"Document: {title_map.get(c['document_name'], c['document_name'])}\n\n{c['chunk_text']}"
        for c in all_chunks
    ]

    embeddings = model.encode(texts_for_embedding, show_progress_bar=True)

    for chunk, embedding in zip(all_chunks, embeddings):
        chunk["embedding"] = embedding

    return all_chunks


def build_embedded_chunks():
    """Processes all PDFs in data/raw/ with title-enriched embeddings."""
    from chunker_langchain import chunk_text_langchain as chunk_text_fn

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    pages = load_all_pdfs()
    print(f"Total chunks to embed: (computing...)")
    return _chunk_and_embed(pages, chunk_text_fn, model)


def build_embedded_chunks_from_files(file_paths):
    """Processes a specific list of uploaded file paths with title-enriched embeddings."""
    from chunker_langchain import chunk_text_langchain as chunk_text_fn

    model = SentenceTransformer(MODEL_NAME)

    pages = []
    for path in file_paths:
        doc_name = os.path.basename(path)
        pages.extend(extract_text_from_pdf(path, doc_name=doc_name))

    return _chunk_and_embed(pages, chunk_text_fn, model)


if __name__ == "__main__":
    embedded_chunks = build_embedded_chunks()

    print(f"\nDone. Total embedded chunks: {len(embedded_chunks)}")
    if embedded_chunks:
        sample = embedded_chunks[0]
        print(f"\n--- Sample chunk ---")
        print(f"Document: {sample['document_name']}, Page: {sample['page_number']}")
        print(f"Text preview: {sample['chunk_text'][:150]}...")
        print(f"Embedding shape: {sample['embedding'].shape}")