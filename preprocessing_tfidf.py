
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# 1. Support tickets
# --------------------------------------------------
tickets = [
    "I was charged twice for my subscription",
    "The app crashes when I open settings",
    "Please reset my password, cannot log in",
    "Refund not received for cancelled order",
    "Error 500 when uploading a file",
    "How do I change my registered email"
]

# --------------------------------------------------
# 2. Preprocessing: lowercase, remove punctuation,
#    and tokenize
# --------------------------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return text, tokens

cleaned_tickets = []
print("PREPROCESSED TICKETS")

for i, ticket in enumerate(tickets, start=1):
    cleaned, tokens = preprocess(ticket)
    cleaned_tickets.append(cleaned)
    print(f"Ticket {i}: {cleaned}")
    print(f"Tokens: {tokens}\n")

# --------------------------------------------------
# 3. TF-IDF features
# --------------------------------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_tickets)

feature_names = vectorizer.get_feature_names_out()

print("TF-IDF MATRIX SHAPE:", tfidf_matrix.shape)

# Five highest-scoring words in ticket 1
ticket1_scores = tfidf_matrix[0].toarray().flatten()
top_indices = ticket1_scores.argsort()[::-1][:5]

print("\nTOP 5 WORDS IN TICKET 1")
for index in top_indices:
    if ticket1_scores[index] > 0:
        print(f"{feature_names[index]}: {ticket1_scores[index]:.4f}")

# --------------------------------------------------
# 4. Find the most similar ticket using cosine similarity
# --------------------------------------------------
query = "cannot log in to my account"
query_vector = vectorizer.transform([query])

similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
best_index = similarities.argmax()

print("\nSIMILARITY RESULT")
print("Query:", query)
print("Most similar ticket:", tickets[best_index])
print("Ticket number:", best_index + 1)
print("Cosine similarity:", round(similarities[best_index], 4))

# --------------------------------------------------
# 5. Create 30 labelled tickets for classification
# --------------------------------------------------
training_tickets = [
    "charged twice for subscription",
    "duplicate payment on my account",
    "subscription billed two times",
    "payment deducted twice",
    "incorrect subscription charge",
    "charged extra money for plan",
    "double billing problem",
    "my card was charged twice",
    "subscription payment issue",
    "unexpected extra charge",

    "app crashes when opening settings",
    "application keeps crashing",
    "settings page closes suddenly",
    "error when opening the app",
    "app freezes on startup",
    "software crashes repeatedly",
    "unable to open settings page",
    "application not responding",
    "app stops working suddenly",
    "technical error in application",

    "cannot log in to my account",
    "please reset my password",
    "forgot my login password",
    "unable to sign in",
    "password reset is not working",
    "login credentials are rejected",
    "cannot access my account",
    "account login failed",
    "need help recovering password",
    "sign in problem"
]

labels = (
    ["billing"] * 10
    + ["technical"] * 10
    + ["login"] * 10
)

# Use stratification so each class is represented in both sets
X_train, X_test, y_train, y_test = train_test_split(
    training_tickets,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

# Fit TF-IDF on training data only to avoid data leakage
classifier_vectorizer = TfidfVectorizer()
X_train_tfidf = classifier_vectorizer.fit_transform(X_train)
X_test_tfidf = classifier_vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

print("\nLOGISTIC REGRESSION RESULTS")
print("Training tickets:", len(X_train))
print("Testing tickets:", len(X_test))
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# --------------------------------------------------
# 6. TF-IDF explanation
# --------------------------------------------------
print("\nWHAT TF-IDF MEASURES")
print("1. Term Frequency (TF) measures how often a word appears in a document.")
print("2. Inverse Document Frequency (IDF) gives less weight to words common across documents.")
print("3. TF-IDF combines both to highlight words important to a particular document.")
