# Ngram Language Model – AIN442 Project 2

This project implements a bigram (and unigram) language model in Python. The model supports tokenization, probability computation, smoothing, and sentence generation based on a training corpus.

## 📚 Course Info
- Course: AIN442 – Practicum in Natural Language Processing
- Semester: 2024–2025 Spring

## 🧠 Project Overview

A class named `ngramLM` is implemented to:

- Learn a unigram and bigram language model from a text corpus
- Provide access to vocabulary and frequency data
- Compute unsmoothed and smoothed probabilities
- Generate new sentences using probabilistic sampling

## 🔧 Class Design

### Instance Variables

- `numOfTokens`: Total number of tokens
- `sizeOfVocab`: Number of unique tokens (vocabulary size)
- `numOfSentences`: Number of sentences
- `sentences`: Tokenized sentences (with `<s>` and `</s>`)
- Unigram and bigram frequency dictionaries

### Key Methods

| Method | Description |
|--------|-------------|
| `trainFromFile(fn)` | Trains the language model on the file `fn` |
| `vocab()` | Returns sorted vocabulary with frequencies |
| `bigrams()` | Returns sorted bigrams with frequencies |
| `unigramCount(word)` | Returns frequency of a unigram |
| `bigramCount((w1,w2))` | Returns frequency of a bigram |
| `unigramProb(word)` | Returns P(word) (unsmoothed) |
| `bigramProb((w1,w2))` | Returns P(w2 | w1) (unsmoothed) |
| `unigramProb_SmoothingUNK(word)` | Add-1 smoothed P(word) |
| `bigramProb_SmoothingUNK((w1,w2))` | Add-1 smoothed P(w2 | w1) |
| `sentenceProb(sent)` | Computes total probability of a sentence |
| `generateSentence(...)` | Generates a random or most probable sentence |

### Tokenization

Sentences are split using a custom regular expression and wrapped with `<s>` and `</s>` tokens. Turkish letters `İ` and `I` are mapped to `i` and `ı` respectively before lowercasing.

## 🧪 Example Usage

```python
lm = ngramLM()
lm.trainFromFile("corpus.txt")
print(lm.unigramProb("bilgisayar"))
print(lm.bigramProb(("bilgisayar", "nedir")))
print(lm.generateSentence(["<s>"], 2, 15))