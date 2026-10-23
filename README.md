# NLP Fundamentals: Preprocessing, TF-IDF and Word Embeddings

## Overview
This project covers fundamental Natural Language Processing (NLP) concepts through two tasks: text preprocessing with TF-IDF and word embeddings using word co-occurrence representations.

## Tasks Included

### Task 1: Preprocessing and TF-IDF
- Convert six support tickets to lowercase.
- Remove punctuation and tokenize the text.
- Extract TF-IDF features using `TfidfVectorizer`.
- Print the five highest-scoring words from the first ticket.
- Find the most similar ticket to "cannot log in to my account" using cosine similarity.
- Train a Logistic Regression classifier on 30 labelled support tickets and calculate accuracy.

### Task 2: Word Embeddings
- Build word representations from a word co-occurrence matrix.
- Find the five nearest words to `king` and `computer`.
- Perform the analogy `king - man + woman`.
- Create sentence vectors by averaging word vectors.
- Compare sentences using cosine similarity.
- Visualize word vectors in two dimensions using PCA.

**Note:** Task 2 uses a co-occurrence matrix and TruncatedSVD rather than pretrained GloVe because the implementation is designed to work without Gensim.

## Technologies Used
- Python 3.14
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Installation

Install the required libraries:

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```

## Project Structure

```text
NLP-Preprocessing-Word-Embeddings/
├── preprocessing_tfidf.py
├── word_embeddings.py
├── README.md
└── word_embeddings_pca.png
```

The PCA image is generated when Task 2 runs successfully.

## How to Run

Run Task 1:

```bash
python preprocessing_tfidf.py
```

Run Task 2:

```bash
python word_embeddings.py
```

## Concepts Covered
- Text preprocessing and tokenization
- TF-IDF vectorization
- Cosine similarity
- Text classification
- Logistic Regression
- Word co-occurrence
- Word embeddings and nearest neighbours
- Word analogies
- Sentence vector representations
- TruncatedSVD and PCA

## Learning Outcomes
This project demonstrates how text can be converted into numerical representations for analysis and classification. It also explores how word and sentence relationships can be measured using vector similarity.

## Author
Dhanush