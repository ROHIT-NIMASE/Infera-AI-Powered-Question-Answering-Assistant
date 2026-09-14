import ollama
from vector_store import get_chroma_client, COLLECTION_NAME

LLM_MODEL_NAME = "llama3.2:3b"


def get_all_chunks_for_document(document_name):
    """
    Retrieves ALL chunks belonging to a specific document, ordered by page number.
    This bypasses similarity search entirely - we want the whole document, not a subset.
    """
    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.get(
        where={"document_name": document_name},
        include=["documents", "metadatas"]
    )

    chunks = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        chunks.append({
            "text": doc,
            "page_number": meta["page_number"]
        })

    # Sort by page number so we process the document in reading order
    chunks.sort(key=lambda c: c["page_number"])
    return chunks
def extract_key_points(document_name, group_size=4):
    """
    Extracts key points/findings from a document using the same
    map-reduce structure as summarization, but focused on bullet points.
    """
    chunks = get_all_chunks_for_document(document_name)

    if not chunks:
        return f"No content found for document: {document_name}"

    # --- MAP step: extract candidate points per group ---
    group_points = []
    for i in range(0, len(chunks), group_size):
        group = chunks[i:i + group_size]
        group_texts = [c["text"] for c in group]
        combined_text = "\n\n".join(group_texts)

        prompt = f"""Extract up to 3 key points (facts, findings, or claims) from the excerpt below.
Return each point as a short bullet starting with "-". If there are no clear key points, return nothing.

Excerpt:
{combined_text}

Key points:"""

        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        group_points.append(response["message"]["content"].strip())

    # --- REDUCE step: deduplicate and consolidate into a final list ---
    combined_points = "\n".join(group_points)
    reduce_prompt = f"""Below are candidate key points extracted from different sections of a paper. 
Consolidate them into a final list of the 5-8 most important, non-redundant key points. 
Return only the bullet list, starting each line with "-". No preamble.

Candidate points:
{combined_points}

Final key points:"""

    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": reduce_prompt}]
    )

    return response["message"]["content"].strip()

def summarize_chunk_group(chunk_texts):
    """Map step: summarizes a small group of chunks into a few sentences."""
    combined_text = "\n\n".join(chunk_texts)

    prompt = f"""Summarize the following excerpt from a research paper in 3-4 sentences. 
Focus on key facts, methods, and findings. Be concise.

Excerpt:
{combined_text}

Summary:"""

    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()


def summarize_document(document_name, group_size=4):
    """
    Full map-reduce summarization:
    1. Get all chunks for the document
    2. Group chunks and summarize each group (map)
    3. Combine group summaries into one final summary (reduce)
    """
    chunks = get_all_chunks_for_document(document_name)

    if not chunks:
        return f"No content found for document: {document_name}"

    print(f"Total chunks for {document_name}: {len(chunks)}")

    # --- MAP step ---
    group_summaries = []
    for i in range(0, len(chunks), group_size):
        group = chunks[i:i + group_size]
        group_texts = [c["text"] for c in group]
        summary = summarize_chunk_group(group_texts)
        group_summaries.append(summary)
        print(f"  Summarized group {i // group_size + 1}/{(len(chunks) - 1) // group_size + 1}")

    # --- REDUCE step ---
    combined_summaries = "\n\n".join(group_summaries)
    reduce_prompt = f"""Below are summaries of different sections of a research paper, in order. 
Combine them into one coherent, well-organized summary of the entire paper (5-7 sentences).
Do not include any preamble like "Here is a summary" or "Here is a X-sentence summary" — output only the summary text itself, starting directly with the content.

Section summaries:
{combined_summaries}

Final summary:"""
    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": reduce_prompt}]
    )

    return response["message"]["content"].strip()

def compare_documents(document_name_1, document_name_2, aspect="methodology and approach"):
    """
    Compares two documents on a given aspect by summarizing each first
    (reusing summarize_document), then asking the LLM to explicitly contrast them.
    """
    print(f"Summarizing {document_name_1}...")
    summary_1 = summarize_document(document_name_1)

    print(f"Summarizing {document_name_2}...")
    summary_2 = summarize_document(document_name_2)

    prompt = f"""Below are summaries of two research papers. Compare them specifically 
in terms of {aspect}. Highlight key similarities and differences. Be specific and concise (4-6 sentences).

Paper A ({document_name_1}):
{summary_1}

Paper B ({document_name_2}):
{summary_2}

Comparison:"""

    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip()
if __name__ == "__main__":
    document_name = "2302.05965v3.pdf"

    print(f"Summarizing: {document_name}\n")
    summary = summarize_document(document_name)
    print(f"\n--- Final Summary ---\n{summary}")

    print(f"\n\nExtracting key points: {document_name}\n")
    key_points = extract_key_points(document_name)
    print(f"\n--- Key Points ---\n{key_points}")
    
    print(f"\n\nComparing documents...\n")
    comparison = compare_documents("2302.05965v3.pdf", "2305.03111v3.pdf", aspect="the text-to-SQL problem they address")
    print(f"\n--- Comparison ---\n{comparison}")