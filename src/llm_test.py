import ollama

MODEL_NAME = "llama3.2:3b"

def ask_llm(prompt):
    """
    Sends a prompt to the local Ollama model and returns the text response.
    """
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]


if __name__ == "__main__":
    question = "Explain what a hash table is in two sentences."
    print(f"Question: {question}\n")

    answer = ask_llm(question)
    print(f"Answer:\n{answer}")