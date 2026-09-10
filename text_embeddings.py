sentences = [
    "Artificial intelligence is transforming the healthcare industry.",
    "Machine learning helps computers learn from data.",
    "Electric vehicles are becoming popular around the world.",
    "Climate change is affecting the environment.",
    "The government introduced a new education policy.",
    "Students use online platforms for learning.",
    "Scientists discovered a new planet in a distant galaxy.",
    "The football team won the championship.",
    "The stock market increased significantly today.",
    "Technology has changed the way people communicate.",
    "Artificial intelligence is changing modern medicine.",
    "Machine learning algorithms can analyze large datasets.",
    "Electric cars can reduce air pollution.",
    "Global warming is causing changes in weather patterns.",
    "A new policy was announced to improve education.",
    "Online learning provides flexible educational opportunities.",
    "Researchers found a previously unknown planet.",
    "The team celebrated after winning the football match.",
    "Share prices rose in the financial market.",
    "Social media has changed human communication."
]

print("Number of sentences:", len(sentences))
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")
embeddings = model.encode(sentences)

print("Embeddings generated successfully!")
print("Embedding shape:", embeddings.shape)
print("\nFirst sentence:")
print(sentences[0])

print("\nFirst sentence embedding:")
print(embeddings[0])
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)

print("\nSimilarity matrix:")
print(similarity_matrix)
pairs = [
    (0, 10),
    (1, 11),
    (2, 12),
    (3, 13),
    (4, 14),
    (5, 15),
    (6, 16),
    (7, 17),
    (8, 18),
    (9, 19)
]

print("\n10 Sentence Pairs and Similarity Scores:\n")

for i, j in pairs:
    score = similarity_matrix[i][j]
    print(f"Pair {i+1}:")
    print("Sentence 1:", sentences[i])
    print("Sentence 2:", sentences[j])
    print(f"Cosine Similarity: {score:.4f}")
    print("-" * 60)
    import pandas as pd

similarity_data = []

for i, j in pairs:
    score = similarity_matrix[i][j]

    similarity_data.append({
        "Pair": f"Pair {i+1}",
        "Sentence 1": sentences[i],
        "Sentence 2": sentences[j],
        "Cosine Similarity": round(float(score), 4)
    })

similarity_df = pd.DataFrame(similarity_data)

similarity_df.to_csv("similarity_scores.csv", index=False)

print("\nSimilarity scores saved successfully!")
print("File name: similarity_scores.csv")
embedding_df = pd.DataFrame(embeddings)

embedding_df.insert(0, "Sentence", sentences)

embedding_df.to_csv("embeddings.csv", index=False)

print("\nEmbeddings saved successfully!")
print("File name: embeddings.csv")
top_5 = similarity_df.sort_values(
    by="Cosine Similarity",
    ascending=False
).head(5)

print("\nTop 5 Most Similar Sentence Pairs:\n")
print(top_5.to_string(index=False))
print("\n--- Observations ---")

for index, row in top_5.iterrows():
    print(f"\n{row['Pair']} - Similarity: {row['Cosine Similarity']}")
    print("Sentence 1:", row["Sentence 1"])
    print("Sentence 2:", row["Sentence 2"])
    print("\n--- Final Observations ---")

print("1. Sentence embeddings represent the meaning of text as numerical vectors.")
print("2. Similar sentences generally have higher cosine similarity scores.")
print("3. Unrelated sentences usually have lower similarity scores.")
print("4. Cosine similarity helps compare the semantic meaning of sentences.")
print("5. Embeddings are useful in semantic search and LLM applications.")