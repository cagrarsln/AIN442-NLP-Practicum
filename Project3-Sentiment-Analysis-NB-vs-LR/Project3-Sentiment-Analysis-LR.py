from datasets import load_dataset
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import re
from collections import Counter
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

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


# 2. Logistic Regression

# 2.1 Training

def bias_scores(train_df):
    pos_counter = Counter()
    neg_counter = Counter()

    for i in range(len(train_df)):
        text = train_df.iloc[i]["text"]
        label = train_df.iloc[i]["label"]
        if label == 0:
            neg_counter.update(text.split())
        else:
            pos_counter.update(text.split())

    total_counter = pos_counter + neg_counter
    all_words = total_counter.keys()

    bias_list = []

    for word in all_words:
        fp = pos_counter[word]
        fn = neg_counter[word]
        ft = fp + fn

        if ft > 0:
            score = abs(((fp - fn) / ft)) * math.log(ft)
            bias_list.append((word, fp, fn, ft, score))

    bias_list_sorted = sorted(bias_list, key=lambda x: (-x[4], x[0]))
    final_list = bias_list_sorted[:10000]
    return final_list

top_10000 = bias_scores(train_df)

vocab = [word for (word, _, _, _, _) in top_10000]
vectorizer = CountVectorizer(vocabulary=vocab)

X_train = vectorizer.transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

y_train = train_df['label']
y_test = test_df['label']


train_accuracies = []
test_accuracies = []
for iter in range(1, 26):
    log_reg = LogisticRegression(max_iter=iter)
    log_reg.fit(X_train, y_train)

    train_acc = log_reg.score(X_train, y_train)

    test_acc = log_reg.score(X_test, y_test)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)


# Plotting graphs (I got help from GPT in this part.)

max_iter_list = list(range(1, 26))

plt.figure(figsize=(10,6))
plt.plot(max_iter_list, train_accuracies, label='Train Accuracy')
plt.plot(max_iter_list, test_accuracies, label='Test Accuracy')
plt.xlabel('Max Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max Iterations (Logistic Regression)')
plt.legend()
plt.grid(True)
plt.show()

# 2.2 Examples
scores = bias_scores(train_df)
print(scores[:2])
print(scores[-2:])

# 3. Analysis
"""
Between the two models, I would prefer Logistic Regression.
In the results, Naive Bayes had about 82% accuracy, while Logistic Regression reached around 86% as the number of iterations increased.

Also, train and test accuracies were close to each other in Logistic Regression, showing that it did not overfit.
Because of the higher accuracy, Logistic Regression would be a better choice for this assignment.
"""