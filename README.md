# Infera — AI-Powered Question Answering Assistant

Infera is a document intelligence and research assistant that lets you upload one or more PDFs (research papers, reports, technical documents) and ask natural-language questions about them. It retrieves relevant passages using semantic search, generates grounded answers with a local LLM, and cites the exact source document and page for every claim — refusing to answer when the information genuinely isn't present rather than guessing.

Built as a hands-on, from-scratch implementation of a full Retrieval-Augmented Generation (RAG) pipeline: no black-box frameworks hiding the core mechanics, and every component (chunking, embeddings, retrieval, reranking, memory, evaluation) implemented and tested individually before being wired together.

---

## Problem Statement

Reading and cross-referencing long technical documents — research papers, specs, reports — is slow, and general-purpose chatbots either can't access private/local documents or confidently hallucinate answers when they don't actually know something. Infera addresses both: it grounds every answer in retrieved source text, cites exactly where each claim came from, and explicitly says when it can't find an answer instead of making one up.

---

## Features

- Multi-PDF upload and processing directly in the browser
- Semantic (meaning-based) search across all uploaded documents
- Source-grounded answers with document name + page number citations
- Honest "I don't know" responses when context doesn't support an answer
- Multi-turn conversation memory (follow-up questions like "what accuracy did *it* achieve?")
- Document-specific filtering ("What methodology did Paper 1 use?")
- Cross-document comparison ("Compare Paper 1 and Paper 2")
- Document summarization and key-point extraction (map-reduce, handles long documents)
- Hybrid retrieval ranking: embedding similarity + cross-encoder reranking, combined via Reciprocal Rank Fusion
- Automated evaluation suite (retrieval relevance + answer correctness) with a hand-curated test set
- Fully local: no API keys, no cloud LLM calls — runs entirely offline via Ollama

---

## Architecture

```
PDF Upload
    ↓
Text Extraction (pypdf, per-page)
    ↓
Chunking (LangChain RecursiveCharacterTextSplitter, page-scoped)
    ↓
Title-Enriched Embeddings (sentence-transformers, all-MiniLM-L6-v2)
    ↓
Vector Store (ChromaDB, persistent, with document/page metadata)
    ↓
User Question
    ↓
Query Rewriting (LLM resolves pronouns/follow-ups using chat history)
    ↓
Document-Filter Detection (routes to specific document(s) if named)
    ↓
Retrieval (top-15 candidate pool, embedding similarity)
    ↓
Reranking (cross-encoder + Reciprocal Rank Fusion → top-5)
    ↓
Grounded Prompt Construction (context + citation instructions)
    ↓
Local LLM Generation (Llama 3.2 3B via Ollama)
    ↓
Answer + Deduplicated Source Citations
```

Comparison and summarization questions route through a separate map-reduce pipeline (see below) rather than the standard retrieval path, since they require reasoning over entire documents rather than a handful of ranked chunks.

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| PDF parsing | `pypdf` | Simple, reliable page-level text + metadata extraction |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Structure-aware splitting with configurable overlap |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Fast, CPU-friendly, well-established baseline |
| Vector DB | ChromaDB | Built-in metadata storage and filtering, simple persistence |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lightweight cross-encoder for precision reranking |
| LLM | Llama 3.2 (3B) via Ollama | Free, fully local, no API costs or key management |
| UI | Streamlit | Fast to build a real chat interface with file upload |
| Language | Python 3.14 | — |

---

## Installation

```bash
git clone <your-repo-url>
cd infera
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com/download) separately, then pull the model:

```bash
ollama pull llama3.2:3b
```

Run the app:

```bash
streamlit run src/app.py
```

---

## How It Works

1. **Upload** one or more PDFs through the sidebar.
2. Each PDF is split page-by-page, then chunked (~1000 characters, 150-character overlap) using structure-aware splitting.
3. Every chunk is embedded — enriched with the source document's title so that even chunks with weak standalone semantic content (e.g. an author list) retain document-level context.
4. Chunks are stored in ChromaDB alongside document name and page number metadata.
5. On a question: the query is rewritten against conversation history (if needed), optionally scoped to a named document, retrieved via embedding similarity (top 15), then reranked via a cross-encoder blended with the original embedding rank (top 5).
6. The top chunks are inserted into a grounding prompt instructing the LLM to answer only from the provided context, or explicitly say it can't find the answer.
7. Comparison questions are detected by keyword + multi-document match and routed to a dedicated map-reduce comparison pipeline instead.

---

## Example Questions

- "What text-to-SQL model architecture does RESDSQL use?"
- "What accuracy did it achieve?" *(follow-up — resolves "it" via conversation memory)*
- "Compare the approaches used in Paper 1 and Paper 2."
- "What methodology was used in Paper 1?"
- "What is the capital of France?" *(correctly refused — out of scope)*

---

## Evaluation

A 14-question hand-curated evaluation set was built covering all uploaded documents, comparison questions, and deliberately out-of-scope questions. Each question was scored automatically on two independent metrics:

| Metric | Result |
|---|---|
| **Retrieval relevance** (correct document retrieved, where applicable) | 11/11 (100%) |
| **Answer correctness** (keyword-containment check) | 10/14 (71.4%) |

**Key finding:** retrieval was reliable in every single test case; the gap in answer correctness traces almost entirely to the generation step (LLM phrasing/caution) and to false negatives in the keyword-matching evaluation script itself, not to retrieval failures. Manual review of the "failed" cases suggests true answer correctness is closer to **~85%** — the evaluation methodology's own limitations (see below) account for much of the measured gap.

One real bug was caught and fixed *because of* this evaluation process: comparison questions were initially falling through to standard single-document retrieval instead of the dedicated comparison pipeline. Adding intent-based routing fixed it, confirmed by re-running the evaluation set.

---

## Challenges & Solutions

- **Vague queries retrieve poorly.** A question like "What methodology was used?" scored far worse than a specific one naming the actual technique. *Mitigation:* reranking + document filtering narrow the search space; documented as an inherent limitation of embedding-based retrieval.
- **A single noisy chunk buried a document's title.** A PDF's title landed in the same chunk as copyright boilerplate, diluting its embedding so badly it ranked 13th out of 15 candidates for a question the title directly answered. *Fix:* every chunk's embedding is now enriched with its document's extracted title before encoding, without altering the chunk text shown to the LLM.
- **Reranking isn't a free win.** The cross-encoder reranker showed a measurable bias toward FAQ-style phrasing over technically-relevant-but-differently-phrased content. *Fix:* combined embedding rank and reranker rank via Reciprocal Rank Fusion instead of trusting the reranker alone.
- **Corrupted local Python environment.** Interrupted installs left `torch` and `numpy` in a broken state (missing package metadata) partway through setup; diagnosed via `pip show` failures and fixed via manual reinstall.
- **No CUDA wheels for Python 3.14 yet.** GPU acceleration wasn't available for this Python version at the time of building; the project runs on CPU, which is fully sufficient at this document scale (a few seconds per embedding batch, a few seconds per generated answer).

---

## Known Limitations

- The small local LLM (Llama 3.2 3B) occasionally produces overly cautious or grammatically awkward answers when retrieved context is only partially relevant, sometimes saying "I could not find this" even when a partial answer is present.
- Document detection recognizes explicit aliases ("Paper 1") and extracted paper titles, but not informal nicknames that differ from a paper's formal title (e.g., a benchmark's common name vs. the paper's actual title).
- Evaluation uses automated keyword-containment matching, which produces false negatives when a correct answer is phrased differently than the expected keywords (e.g., "identifies" vs. "identify").
- Answer generation is not fully deterministic between runs on identical input, a known characteristic of LLM sampling; partially mitigated with a lowered temperature setting.

---

## Future Improvements

- LLM-based extraction of document nicknames/common names (not just formal titles) for more robust document-filtering
- Hybrid search combining embeddings with keyword/BM25 matching for exact-term queries
- LLM-as-judge evaluation to complement keyword-based scoring
- Swap in a larger/hosted LLM for improved answer consistency, with local model retained as a free-tier fallback
- OCR support for scanned PDFs with no embedded text layer

---

## Project Structure

```
infera/
├── src/
│   ├── app.py                  # Streamlit UI
│   ├── pdf_loader.py           # PDF text extraction
│   ├── chunker_langchain.py    # Text chunking
│   ├── embedder.py             # Title-enriched embedding generation
│   ├── vector_store.py         # ChromaDB storage/retrieval
│   ├── document_filter.py      # Document-name/alias detection
│   ├── reranker.py             # Cross-encoder + RRF blended reranking
│   ├── query_rewriter.py       # Conversation-memory query rewriting
│   ├── summarizer.py           # Map-reduce summarization/comparison
│   ├── rag_pipeline.py         # Main orchestration + entry point
│   └── evaluator.py            # Automated evaluation runner
├── data/
│   ├── raw/                    # Uploaded/source PDFs
│   ├── chroma_db/              # Persistent vector store
│   └── eval_dataset.py         # Evaluation questions + expected answers
├── requirements.txt
└── README.md
```