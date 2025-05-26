# Sentiment Analysis with Naive Bayes & Logistic Regression – AIN442 Project 3

This project implements two text classification models to perform binary sentiment classification (positive vs. negative) on the IMDB Movie Reviews dataset. It compares the performance of Naive Bayes and Logistic Regression approaches.

## 📚 Course Info
- Course: AIN442 – Practicum in Natural Language Processing
- Semester: 2024–2025 Spring

## 🧠 Project Overview

The dataset is loaded via `load_dataset("imdb")` and split into `train_df` and `test_df`.

### ✂️ Preprocessing Steps
Each text sample is cleaned using:
- Punctuation removal
- Digit removal
- Lowercasing
- Stopword removal (from `nltk.corpus.stopwords`)
- Extra whitespace cleanup

All preprocessing is handled by the `preprocess_text(text)` function used in both models.

---

## 📌 Part 1 – Naive Bayes Classifier (`Project3-Sentiment-Analysis-NB.py`)

Custom `NaiveBayesClassifier` class implements:

- Word frequency counting by class
- Log-probability computation with add-one smoothing
- Class prediction for new texts

### Attributes:
- `prior_pos`, `prior_neg`
- `total_pos_words`, `total_neg_words`
- `pos_counter`, `neg_counter`, `vocab_size`

### Output:
Predicts sentiment and prints log probabilities for both classes.

---

## 📌 Part 2 – Logistic Regression (`Project3-Sentiment-Analysis-LR.py`)

Steps:
1. Calculate **bias scores** of all words (favoring highly class-indicative words)
2. Select top 10,000 most biased words
3. Build Bag-of-Words vectors using `CountVectorizer`
4. Train `LogisticRegression` from `sklearn` for `max_iter = 1...25`
5. Plot train vs. test accuracy using `matplotlib.pyplot`

Final section includes a discussion of the preferred model based on accuracy trends.

---

## ▶️ How to Run

**Naive Bayes**:
```bash
python Project3-Sentiment-Analysis-NB.py