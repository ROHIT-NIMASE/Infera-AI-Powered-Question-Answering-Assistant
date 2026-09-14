from vector_store import get_chroma_client, COLLECTION_NAME


def get_all_document_names():
    """Returns a sorted list of unique document names currently stored in Chroma."""
    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)
    all_data = collection.get(include=["metadatas"])
    doc_names = sorted(set(m["document_name"] for m in all_data["metadatas"]))
    return doc_names


def get_document_title_keywords():
    """
    For each document, grabs its Page 1 text and extracts significant
    keywords (likely from the title) to use for natural-language matching.
    Returns: {doc_name: [keyword1, keyword2, ...]}
    """
    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)
    all_data = collection.get(include=["metadatas", "documents"])

    # Find each document's first page_number == 1 chunk
    first_page_text = {}
    for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
        if meta["page_number"] == 1 and meta["document_name"] not in first_page_text:
            first_page_text[meta["document_name"]] = doc_text

    keywords_map = {}
    for doc_name, text in first_page_text.items():
        title_line = text.strip().split("\n")[0][:80]  # first line, first 80 chars ~ title
        words = title_line.replace(":", " ").replace(",", " ").split()
        # Keep only meaningful words: 4+ letters, alphabetic only (skip numbers/symbols)
        significant = [w.lower() for w in words if len(w) >= 4 and w.isalpha()]
        keywords_map[doc_name] = significant

    return keywords_map


def build_document_alias_map():
    """
    Maps human-friendly aliases to actual filenames:
    - 'paper 1', 'paper 2', ... (by upload order)
    - title keywords extracted from each document's first page
    """
    doc_names = get_all_document_names()
    alias_map = {}
    for i, name in enumerate(doc_names, start=1):
        alias_map[f"paper {i}"] = name
        alias_map[f"paper{i}"] = name

    keywords_map = get_document_title_keywords()
    for doc_name, keywords in keywords_map.items():
        for kw in keywords:
            # Don't overwrite an existing alias if two docs share a common word
            if kw not in alias_map:
                alias_map[kw] = doc_name

    return alias_map, doc_names


def detect_document_filter(query):
    """
    Checks the query for mentions of a specific document, via alias,
    filename substring, or title keyword. Returns a list of matched
    document names (empty list = no filter, search all).
    """
    query_lower = query.lower()
    alias_map, doc_names = build_document_alias_map()

    matched = set()

    for alias, doc_name in alias_map.items():
        if alias in query_lower:
            matched.add(doc_name)

    for doc_name in doc_names:
        name_without_ext = doc_name.replace(".pdf", "").lower()
        if name_without_ext in query_lower:
            matched.add(doc_name)

    return list(matched)


if __name__ == "__main__":
    print("Documents currently stored:")
    for name in get_all_document_names():
        print(f" - {name}")

    print("\nExtracted title keywords per document:")
    for doc_name, kws in get_document_title_keywords().items():
        print(f"  {doc_name}: {kws}")

    alias_map, _ = build_document_alias_map()
    print("\nFull alias map:")
    for alias, name in alias_map.items():
        print(f"  {alias} -> {name}")

    test_queries = [
        "What dataset was used in the RESDSQL paper?",
        "What methodology was used in Paper 1?",
        "Tell me about the BIRD benchmark",
        "Summarize the findings overall",
    ]

    print("\nDetection tests:")
    for q in test_queries:
        result = detect_document_filter(q)
        print(f"  \"{q}\" -> {result}")