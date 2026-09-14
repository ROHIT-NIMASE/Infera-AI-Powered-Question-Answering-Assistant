import sys
sys.path.append('src')
from query_rewriter import rewrite_query
from rag_pipeline import retrieve_chunks

history = [
    {"question": "Which paper introduced the Transformer architecture?",
     "answer": "The Transformer architecture was introduced by Vaswani et al."}
]

follow_up = "What was its main innovation?"
rewritten = rewrite_query(follow_up, history)
print(f"Rewritten query: {rewritten}\n")

chunks = retrieve_chunks(rewritten, top_k=5)
for i, c in enumerate(chunks, start=1):
    print(f"\n[{i}] {c['document_name']}, Page {c['page_number']}")
    print(c['text'][:300])