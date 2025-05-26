import random
import numpy as np
import gensim.downloader

model = gensim.downloader.load("word2vec-google-news-300")

def replace_with_similar(sentence, indices):
    similar_words_dict = {}
    words = sentence.split()

    # get top 5 similar words
    for i in indices:
        word = words[i]
        similar_words = model.most_similar(word, topn=5)
        similar_words_dict[word] = similar_words

    # replace with random similar word
    for i in indices:
        word = words[i]
        similar_words = similar_words_dict[word]
        new_word = random.choice(similar_words)[0]
        words[i] = new_word
    
    new_sentence = " ".join(words)

    return new_sentence, similar_words_dict

def sentence_vector(sentence):
    vector_dict = {}

    words = sentence.split()
    
    # get vector for each word
    for word in words:
        if word in model:
            vector_dict[word] = model[word]
        else:
            vector_dict[word] = np.zeros(300)

    # calculate average vector
    sentence_vec = np.mean(list(vector_dict.values()), axis=0)

    return vector_dict, sentence_vec

def most_similar_sentences(file_path, query):
    with open(file_path, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file]
    
    # get query vector
    _, query_vec = sentence_vector(query)

    similarity_list = []

     # comparing
    for sentence in sentences:
        _, sentence_vec = sentence_vector(sentence)

        # compute cosine similarity
        dot_product = np.dot(sentence_vec, query_vec)
        norms = np.linalg.norm(sentence_vec) * np.linalg.norm(query_vec)
        
        if norms != 0:
            cos_sim = dot_product / norms
        else:
            cos_sim = 0.0
            
        similarity_list.append((sentence, cos_sim))
    
    return sorted(similarity_list, key=lambda x: x[1], reverse=True)



