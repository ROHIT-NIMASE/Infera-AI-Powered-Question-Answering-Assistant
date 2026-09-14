"""
Small hand-curated evaluation dataset for RAG evaluation.
Each entry includes the question, the expected correct answer (for reference),
and which document/page(s) the answer should come from (ground truth for retrieval).
"""

EVAL_DATASET = [
    {
        "id": "eval_01",
        "question": "What text-to-SQL model architecture does RESDSQL use?",
        "expected_answer_contains": ["ranking-enhanced encoder", "skeleton-aware decoder"],
        "expected_document": "2302.05965v3.pdf",
    },
    {
        "id": "eval_02",
        "question": "What dataset was used in the RESDSQL paper?",
        "expected_answer_contains": ["Spider"],
        "expected_document": "2302.05965v3.pdf",
    },
    {
        "id": "eval_03",
        "question": "What is the capital of France?",
        "expected_answer_contains": ["could not find", "don't know", "not present"],
        "expected_document": None,
    },
    {
        "id": "eval_04",
        "question": "Who funded the creation of the BIRD dataset?",
        "expected_answer_contains": ["Alibaba", "DAMO"],
        "expected_document": "2305.03111v3.pdf",
    },
    {
        "id": "eval_05",
        "question": "What methodology was used in Paper 1?",
        "expected_answer_contains": ["schema linking", "skeleton"],
        "expected_document": "2302.05965v3.pdf",
    },
    {
        "id": "eval_06",
        "question": "What are the three principal kinds of data dependencies discussed by Codd?",
        "expected_answer_contains": [
            "ordering dependence",
            "indexing dependence",
            "access path dependence"
        ],
        "expected_document": "codd.pdf",
    },
    {
        "id": "eval_07",
        "question": "What is a primary key in the relational model?",
        "expected_answer_contains": [
            "uniquely identify",
            "n-tuple"
        ],
        "expected_document": "codd.pdf",
    },
    {
        "id": "eval_08",
        "question": "What is normalization and why does Codd introduce it?",
        "expected_answer_contains": [
            "nonsimple domains",
            "eliminating",
            "normal form"
        ],
        "expected_document": "codd.pdf",
    },
    {
        "id": "eval_09",
        "question": "What are the properties of an n-ary relation described by Codd?",
        "expected_answer_contains": [
            "rows",
            "distinct",
            "columns"
        ],
        "expected_document": "codd.pdf",
    },
    {
        "id": "eval_10",
        "question": "What advantages does the relational model provide over tree or network models?",
        "expected_answer_contains": [
            "data independence",
            "natural structure",
            "redundancy",
            "consistency"
        ],
        "expected_document": "codd.pdf",
    },
    {
        "id": "eval_11",
        "question": "Which operations does Codd identify as an adequate collection for deriving relations?",
        "expected_answer_contains": [
            "projection",
            "natural join",
            "restriction"
        ],
        "expected_document": "codd.pdf",
    },
    {
        "id": "eval_12",
        "question": "Compare the approaches used in Paper 1 and Paper 2.",
        "expected_answer_contains": ["schema linking", "benchmark"],
        "expected_document": None,
    },
    {
        "id": "eval_13",
        "question": "What accuracy did RESDSQL achieve on the Spider dataset?",
        "expected_answer_contains": ["EM", "EX"],
        "expected_document": "2302.05965v3.pdf",
    },
    {
        "id": "eval_14",
        "question": "What is the tallest mountain in the world?",
        "expected_answer_contains": ["could not find", "don't know", "not present"],
        "expected_document": None,
    },
]