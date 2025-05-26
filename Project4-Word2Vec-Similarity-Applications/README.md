# Word2Vec Similarity Applications – AIN442 Project 4

This project explores two core applications of word embeddings using the `word2vec-google-news-300` pre-trained model via the Gensim library. It focuses on both **word-level replacement** and **sentence-level similarity**.

## 📚 Course Info
- Course: AIN442 – Practicum in Natural Language Processing
- Semester: 2024–2025 Spring

## 🧠 Project Objectives

### 1. Replace with Similar Words

- **Function**: `replace_with_similar(sentence: str, indices: list)`
- **Purpose**: Randomly replace words in a sentence (at given indices) with similar alternatives using the Word2Vec model.
- **Returns**: 
  - A modified sentence with replaced words
  - A dictionary mapping each replaced word to its top 5 similar words and their similarity scores

**Example Usage**:
```python
new_sentence, similar_dict = replace_with_similar("I love AIN442 and BBM497 courses", [1, 5])