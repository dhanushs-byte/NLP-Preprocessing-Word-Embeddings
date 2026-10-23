
import re
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# --------------------------------------------------
# 1. Sample corpus for learning word relationships
# --------------------------------------------------
sentences = [
    "king is a male royal ruler",
    "queen is a female royal ruler",
    "king and queen rule a kingdom",
    "man is an adult male person",
    "woman is an adult female person",
    "boy is a young male person",
    "girl is a young female person",
    "father is a male parent",
    "mother is a female parent",
    "brother is a male sibling",
    "sister is a female sibling",
    "computer processes digital data",
    "software runs on a computer",
    "hardware and software are computer technology",
    "internet connects computers around the world",
    "technology includes computers and software",
    "python and java are programming languages",
    "programming means writing computer code",
    "data is processed by computer software",
    "apple orange and banana are fruit",
    "fruit and food are important",
    "car bus and train are transportation",
    "a bus travels on the road",
    "a train travels on railway tracks",
    "travel means going to another place",
    "the king married the queen",
    "the man and woman are people",
    "a boy and girl are children",
    "computer code uses programming languages",
    "software processes data using a computer"
]

# --------------------------------------------------
# 2. Tokenization
# --------------------------------------------------
tokenized = [
    re.findall(r"\b[a-z]+\b", sentence.lower())
    for sentence in sentences
]

vocabulary = sorted(set(
    word for sentence in tokenized for word in sentence
))

word_to_id = {word: i for i, word in enumerate(vocabulary)}
vocab_size = len(vocabulary)

print("Vocabulary size:", vocab_size)

# --------------------------------------------------
# 3. Build a word co-occurrence matrix
# --------------------------------------------------
window_size = 2
co_matrix = np.zeros((vocab_size, vocab_size), dtype=float)

for tokens in tokenized:
    for i, word in enumerate(tokens):
        word_id = word_to_id[word]

        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)

        for j in range(start, end):
            if i != j:
                context_id = word_to_id[tokens[j]]
                co_matrix[word_id, context_id] += 1

# Reduce dimensions to obtain dense word representations
dimensions = min(20, vocab_size - 1)

svd = TruncatedSVD(n_components=dimensions, random_state=42)
word_vectors = svd.fit_transform(co_matrix)
word_vectors = normalize(word_vectors)

def nearest_words(word, top_n=5):
    if word not in word_to_id:
        print(f"'{word}' is not in the vocabulary.")
        return

    vector = word_vectors[word_to_id[word]].reshape(1, -1)
    scores = cosine_similarity(vector, word_vectors)[0]
    scores[word_to_id[word]] = -1

    indices = np.argsort(scores)[::-1][:top_n]

    for index in indices:
        print(f"{vocabulary[index]}: {scores[index]:.4f}")

# --------------------------------------------------
# 4. Five nearest words
# --------------------------------------------------
print("\nNearest words to king:")
nearest_words("king")

print("\nNearest words to computer:")
nearest_words("computer")

# --------------------------------------------------
# 5. Word analogy
# king - man + woman
# --------------------------------------------------
print("\nAnalogy: king - man + woman")

if all(word in word_to_id for word in ["king", "man", "woman"]):
    analogy_vector = (
        word_vectors[word_to_id["king"]]
        - word_vectors[word_to_id["man"]]
        + word_vectors[word_to_id["woman"]]
    ).reshape(1, -1)

    scores = cosine_similarity(analogy_vector, word_vectors)[0]

    for word in ["king", "man", "woman"]:
        scores[word_to_id[word]] = -1

    results = np.argsort(scores)[::-1][:5]

    for index in results:
        print(f"{vocabulary[index]}: {scores[index]:.4f}")

    print("Top result:", vocabulary[results[0]])
    print("Note: a small corpus may not return queen.")

# --------------------------------------------------
# 6. Average word vectors to create sentence vectors
# --------------------------------------------------
def sentence_vector(sentence):
    tokens = re.findall(r"\b[a-z]+\b", sentence.lower())
    vectors = [
        word_vectors[word_to_id[word]]
        for word in tokens if word in word_to_id
    ]

    if not vectors:
        return np.zeros(word_vectors.shape[1])

    return np.mean(vectors, axis=0)

sentence1 = "king and queen rule a kingdom"
sentence2 = "man and woman are people"

v1 = sentence_vector(sentence1).reshape(1, -1)
v2 = sentence_vector(sentence2).reshape(1, -1)

score = cosine_similarity(v1, v2)[0, 0]

print("\nSentence 1:", sentence1)
print("Sentence 2:", sentence2)
print(f"Cosine similarity: {score:.4f}")

# --------------------------------------------------
# 7. Bonus: Plot 30 word vectors using PCA
# --------------------------------------------------
selected_words = [
    word for word in [
        "king", "queen", "man", "woman", "boy", "girl",
        "father", "mother", "brother", "sister",
        "computer", "software", "hardware", "internet",
        "technology", "python", "java", "programming",
        "code", "data", "apple", "orange", "banana",
        "fruit", "food", "car", "bus", "train", "road",
        "travel"
    ] if word in word_to_id
]

selected_vectors = np.array([
    word_vectors[word_to_id[word]] for word in selected_words
])

pca = PCA(n_components=2)
points = pca.fit_transform(selected_vectors)

plt.figure(figsize=(12, 8))
plt.scatter(points[:, 0], points[:, 1])

for i, word in enumerate(selected_words):
    plt.annotate(word, (points[i, 0], points[i, 1]))

plt.title("Word Co-occurrence Embeddings with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("word_embeddings_pca.png", dpi=300)
plt.show()

print("\nPCA plot saved as word_embeddings_pca.png")
print("Words plotted:", len(selected_words))
