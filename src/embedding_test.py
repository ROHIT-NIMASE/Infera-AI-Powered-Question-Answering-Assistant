from sentence_transformers import SentenceTransformer, util

# Load the pre-trained embedding model (downloads once, then caches locally)
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat.",
    "A dog is barking loudly.",
    "The model achieved 94.2% accuracy on the test set.",
    "What accuracy did the model achieve?",
]

embeddings = model.encode(sentences)

print(f"Number of sentences: {len(sentences)}")
print(f"Embedding shape: {embeddings.shape}")  # (4, 384) expected
print(f"First 10 values of embedding for sentence 1:\n{embeddings[0][:10]}\n")

# Compare similarity between the question and each sentence
question_embedding = embeddings[3]
for i in range(3):
    similarity = util.cos_sim(question_embedding, embeddings[i])
    print(f"Similarity between question and sentence {i+1}: {similarity.item():.4f}")
    print(f"  -> \"{sentences[i]}\"")