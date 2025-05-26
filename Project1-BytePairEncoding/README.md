# Byte Pair Encoding (BPE) Tokenizer – AIN442 Project 1

This project implements a variation of the Byte-Pair Encoding (BPE) algorithm for subword tokenization in Python using only built-in libraries (`re`, `codecs`). The implementation consists of a token learner and a token segmenter, both built from scratch without external NLP libraries.

## 📚 Course Info
- Course: AIN442 – Practicum in Natural Language Processing
- Semester: 2024–2025 Spring

## 🧠 Project Objective

To learn subword-level tokens from a training corpus using a modified BPE algorithm and use the learned merges to tokenize new texts. This approach improves vocabulary coverage and helps handle out-of-vocabulary words.

## 🔧 Implemented Components

### 1. Token Learner
- `bpeCorpus(corpus: str, maxMergeCount=10)`
- `bpeFN(filename: str, maxMergeCount=10)`
  
These functions:
- Parse the training corpus into character-level token lists
- Apply BPE merges to create a token vocabulary
- Return: `Merges`, `Vocabulary`, `TokenizedCorpus`

### 2. Token Segmenter
- `bpeTokenize(input_str: str, merges: list)`

This function:
- Applies the learned merges to a new input string
- Returns tokenized text as subword units

### 3. Output File Writer
- `bpeFNToFile(input_fn, maxMergeCount=10, output_fn="output.txt")`

Writes the `Merges`, `Vocabulary`, and `TokenizedCorpus` to a file.

## 📄 Example Output

Sample input:
```python
(Merges, Vocab, TokenizedCorpus) = bpeCorpus("sos ses sus", maxMergeCount=6)