from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],  # priority order, same idea as your fallback logic
    )


def chunk_text_langchain(text):
    """
    Splits text using LangChain's RecursiveCharacterTextSplitter.
    Returns a list of chunk strings — same output shape as your hand-rolled chunk_text().
    """
    splitter = get_splitter()
    return splitter.split_text(text)


if __name__ == "__main__":
    sample_text = (
        "This is the first paragraph. It talks about an introduction to the topic "
        "and sets up the background context for the reader.\n\n"
        "This is the second paragraph. It goes deeper into the methodology used "
        "in the study, explaining the dataset and the model architecture chosen.\n\n"
        "This is the third paragraph. It discusses the results obtained and how "
        "they compare to prior work in the field.\n\n"
        "This is the fourth and final paragraph. It wraps up with limitations "
        "and directions for future research."
    )

    chunks = chunk_text_langchain(sample_text)
    # Use small size/overlap to match your earlier hand-rolled test
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_text(sample_text)

    print(f"Number of chunks: {len(chunks)}\n")
    for i, c in enumerate(chunks, start=1):
        print(f"--- Chunk {i} ({len(c)} chars) ---")
        print(c)
        print()