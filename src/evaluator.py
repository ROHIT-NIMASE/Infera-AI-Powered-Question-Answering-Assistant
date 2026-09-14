import sys
import os
sys.path.append(os.path.dirname(__file__))
from rag_pipeline import handle_query

# Import the eval dataset - adjust path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from eval_dataset import EVAL_DATASET


def check_answer_correctness(answer, expected_keywords):
    """
    Checks if ANY of the expected keywords appear in the answer (case-insensitive).
    Using ANY (not ALL) since a correct answer might use only some expected terms.
    """
    answer_lower = answer.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    return len(matched) > 0, matched


def check_retrieval_relevance(sources, expected_document):
    """
    Checks if the expected document appears among the retrieved sources.
    If expected_document is None (out-of-scope or multi-doc question), this check is skipped.
    """
    if expected_document is None:
        return None  # Not applicable
    source_docs = [s["document_name"] for s in sources]
    return expected_document in source_docs


def run_evaluation():
    results = []

    for item in EVAL_DATASET:
        print(f"Running: {item['id']} - {item['question']}")

        answer, sources, _ = handle_query(item["question"], chat_history=[])

        correctness, matched_keywords = check_answer_correctness(
            answer, item["expected_answer_contains"]
        )
        retrieval_ok = check_retrieval_relevance(sources, item["expected_document"])

        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "answer_correct": correctness,
            "matched_keywords": matched_keywords,
            "retrieval_correct": retrieval_ok,
            "sources": [f"{s['document_name']} p{s['page_number']}" for s in sources],
        })

    return results


def print_report(results):
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    correct_count = sum(1 for r in results if r["answer_correct"])
    total = len(results)

    retrieval_checks = [r for r in results if r["retrieval_correct"] is not None]
    retrieval_correct_count = sum(1 for r in retrieval_checks if r["retrieval_correct"])

    for r in results:
        status = "PASS" if r["answer_correct"] else "FAIL"
        retrieval_status = (
            "N/A" if r["retrieval_correct"] is None
            else ("PASS" if r["retrieval_correct"] else "FAIL")
        )
        print(f"\n[{r['id']}] Answer: {status} | Retrieval: {retrieval_status}")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer'][:150]}...")
        print(f"  Matched keywords: {r['matched_keywords']}")
        print(f"  Sources: {r['sources']}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: Answer correctness: {correct_count}/{total} ({100*correct_count/total:.1f}%)")
    print(f"SUMMARY: Retrieval relevance: {retrieval_correct_count}/{len(retrieval_checks)} "
          f"({100*retrieval_correct_count/len(retrieval_checks):.1f}%) [where applicable]")
    print("=" * 70)


if __name__ == "__main__":
    results = run_evaluation()
    print_report(results)