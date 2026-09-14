CHUNK_SIZE = 1000      # max characters per chunk
CHUNK_OVERLAP = 150    # characters repeated between consecutive chunks


def split_into_paragraphs(text):
    """Split text on blank lines into a list of paragraph strings."""
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]  # drop empty strings


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits a block of text into overlapping chunks, preferring to
    break on paragraph boundaries. Falls back to raw character
    slicing if a single paragraph exceeds chunk_size.
    Returns a list of chunk strings.
    """
    paragraphs = split_into_paragraphs(text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If the paragraph itself is too big, split it by raw characters
        if len(para) > chunk_size:
            # First, flush whatever we've built up so far
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start = end - overlap  # step back by overlap for next slice

            continue

        # Would adding this paragraph exceed our limit?
        if len(current_chunk) + len(para) + 2 > chunk_size:  # +2 for "\n\n"
            chunks.append(current_chunk)
            # Start new chunk, seeded with overlap from the end of the last one
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


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

    chunks = chunk_text(sample_text, chunk_size=200, overlap=40)

    print(f"Number of chunks: {len(chunks)}\n")
    for i, c in enumerate(chunks, start=1):
        print(f"--- Chunk {i} ({len(c)} chars) ---")
        print(c)
        print()