import ollama

LLM_MODEL_NAME = "llama3.2:3b"


def rewrite_query(current_question, chat_history):
    """
    Uses the LLM to rewrite a potentially ambiguous follow-up question
    into a standalone question, using prior conversation turns as context.

    chat_history: list of dicts like [{"question": ..., "answer": ...}, ...]
    Returns: a rewritten, standalone question string.
    """
    if not chat_history:
        # No prior context exists yet, so there's nothing to resolve against.
        return current_question

    history_text = ""
    for turn in chat_history:
        history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n\n"

    prompt = f"""Given the conversation history below, rewrite the follow-up question 
to be a standalone question that includes all necessary context. 
Do not answer the question — only rewrite it. 
If the follow-up question is already standalone and doesn't depend on the history, return it unchanged.

Conversation history:
{history_text}

Follow-up question: {current_question}

Standalone question:"""

    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    rewritten = response["message"]["content"].strip()
    return rewritten


if __name__ == "__main__":
    history = [
        {"question": "What dataset was used in the RESDSQL paper?",
         "answer": "The RESDSQL paper used the Spider dataset and its variants."}
    ]

    follow_up = "What accuracy did they achieve?"

    rewritten = rewrite_query(follow_up, history)
    print(f"Original: {follow_up}")
    print(f"Rewritten: {rewritten}")