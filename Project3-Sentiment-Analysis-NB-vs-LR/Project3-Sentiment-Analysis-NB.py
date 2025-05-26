from datasets import load_dataset
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import string
from collections import Counter
import math
from sklearn.metrics import accuracy_score


# Dataset 

# Loading datasets

dataset = load_dataset("imdb")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# 1. Preprocessing

def remove_punctuation(text):
    new_text = ""

    for char in text:
        if char in string.punctuation:
            new_text += " "
        else:
            new_text += char

    return new_text

def remove_numbers(text):
    new_text = ""

    for char in text:
        if char in "0123456789":
            new_text += " "
        else:
            new_text += char

    return new_text

def convert_lowercase(text):
    return text.lower()


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    cleared_words = []

    for word in words:
        if word in stop_words:
            cleared_words.append(" ")
        else:
            cleared_words.append(word)

    final_text = " ".join(cleared_words)

    return final_text

def clean_spaces(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def preprocess_text(text):
    first_step = remove_punctuation(text)
    second_step = remove_numbers(first_step)
    third_step = convert_lowercase(second_step)
    fourth_step = remove_stopwords(third_step)
    fifth_step = clean_spaces(fourth_step)
    return fifth_step

train_df["text"] = train_df["text"].apply(preprocess_text)
test_df["text"] = test_df["text"].apply(preprocess_text)

# Example for preprocessing
example_text = "Hello world!!! I love the <AIN442> and <BBM497> courses."
first_step = remove_punctuation(example_text)
second_step = remove_numbers(first_step)
third_step = convert_lowercase(second_step)
fourth_step = remove_stopwords(third_step)
fifth_step = clean_spaces(fourth_step)

print(first_step)
print(second_step)
print(third_step)
print(fourth_step)
print(fifth_step)

# Naive Bayes

# 2.1 Training

class NaiveBayesClassifier:
    def __init__(self):
        self.total_pos_words = 0
        self.total_neg_words = 0
        self.vocab_size = 0
        self.prior_pos = 0
        self.prior_neg = 0
        self.pos_counter = Counter()
        self.neg_counter = Counter()

    def fit(self, train_df):
        pos_samples = train_df[train_df["label"] == 1]
        neg_samples = train_df[train_df["label"] == 0]

        self.prior_pos = len(pos_samples) / len(train_df)
        self.prior_neg = len(neg_samples) / len(train_df)

        self.pos_counter = Counter()
        self.neg_counter = Counter()

        for text in pos_samples['text']:
            self.pos_counter.update(text.split())

        for text in neg_samples['text']:
            self.neg_counter.update(text.split())

        self.total_pos_words = sum(self.pos_counter.values())
        self.total_neg_words = sum(self.neg_counter.values())

        all_unique_words = set(list(self.pos_counter.keys()) + list(self.neg_counter.keys()))
        self.vocab_size = len(all_unique_words)

    def predict(self, text):
        # Preprocessing
        text = preprocess_text(text)

        log_prob_pos = math.log(self.prior_pos)
        log_prob_neg = math.log(self.prior_neg)

        # Collect probabilities
        for word in text.split():
            # positive class
            count_w_pos = self.pos_counter.get(word, 0)
            prob_w_pos = (count_w_pos + 1) / (self.total_pos_words + self.vocab_size)
            log_prob_pos += math.log(prob_w_pos)

            # negative class
            count_w_neg = self.neg_counter.get(word, 0)
            prob_w_neg = (count_w_neg + 1) / (self.total_neg_words + self.vocab_size)
            log_prob_neg += math.log(prob_w_neg)

        # Comparing

        prediction = 0

        if log_prob_pos > log_prob_neg:
            prediction = 1

        return prediction, log_prob_pos, log_prob_neg

# 2.2 Testing with Examples
nb = NaiveBayesClassifier()
nb.fit(train_df)

print(nb.total_pos_words)
print(nb.total_neg_words)
print(nb.vocab_size)
print(nb.prior_pos)
print(nb.prior_neg)
print(nb.pos_counter["great"])
print(nb.neg_counter["great"])

prediction1 = nb.predict(test_df.iloc[0]["text"])

prediction2 = nb.predict("This movie will be place at 1st in my favourite movies!")

prediction3 = nb.predict("I couldn’t wait for the movie to end, so I turned it off halfway through. :D It was a complete disappointment.")

print('Positive' if prediction1[0] == 1 else 'Negative')
print(prediction1)

print('Positive' if prediction2[0] == 1 else 'Negative')
print(prediction2)

print('Positive' if prediction3[0] == 1 else 'Negative')
print(prediction3)

# Test with test_df
y_true = test_df['label'].values

y_pred = [nb.predict(text)[0] for text in test_df['text']]

accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")

# 3. Analysis
"""
Between the two models, I would prefer Logistic Regression.
In the results, Naive Bayes had about 82% accuracy, while Logistic Regression reached around 86% as the number of iterations increased.

Also, train and test accuracies were close to each other in Logistic Regression, showing that it did not overfit.
Because of the higher accuracy, Logistic Regression would be a better choice for this assignment.
"""