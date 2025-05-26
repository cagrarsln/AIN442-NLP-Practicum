# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 15:06:51 2025

"""


import random
import re


# ngramLM CLASS
class ngramLM:
    """Ngram Language Model Class"""
    
    # Create Empty ngramLM
    def __init__(self):
        self.numOfTokens = 0
        self.sizeOfVocab = 0
        self.numOfSentences = 0
        self.sentences = []
        # TO DO 
        self.unigramFreq = dict()
        self.bigramFreq = dict()

    # INSTANCE METHODS
    def trainFromFile(self,fn):
        # TO DO
        with open(fn, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        pattern = r"""(?x)
        (?:[A-ZÇĞIİÖŞÜ]\.)+              
        | \d+(?:\.\d*)?(?:\'\w+)?   
        | \w+(?:-\w+)*(?:\'\w+)?  
        | \.\.\.  
        | [][,;.?():_!#^+$%&><|/{()=}\"\'\\\"\`-] 
        """

        for line in lines:
            tokens = re.findall(pattern , line)
            
            tokens = [token.replace("I", "ı").replace("İ", "i") for token in tokens]
            tokens = [token.lower() for token in tokens]

            sentence = []
            for token in tokens:
                sentence.append(token)
                if token in [".", "?", "!"]:
                    self.sentences.append(["<s>"] + sentence + ["</s>"])
                    sentence = []
            
            if sentence:
                self.sentences.append(["<s>"] + sentence + ["</s>"])
        
        for sent in self.sentences:
            for i in range(len(sent)):
                word1 = sent[i]
                if word1 not in self.unigramFreq:
                    self.unigramFreq[word1] = 0
                self.unigramFreq[word1] += 1

                if i < len(sent) - 1:
                    word2 = sent[i+1]
                    bigram = (word1, word2)

                    if bigram not in self.bigramFreq:
                        self.bigramFreq[bigram] = 0
                    self.bigramFreq[bigram] += 1
        
        self.numOfTokens = sum(self.unigramFreq.values())
        self.numOfSentences = len(self.sentences)
        self.sizeOfVocab = len(self.unigramFreq)

        
    def vocab(self):
        # TO DO
        vocab_list = list(self.unigramFreq.items())
        vocab_list.sort(key=lambda x: x[0])      # ascending according to word first
        vocab_list.sort(key=lambda x: x[1], reverse=True)   # then decreasing in frequency

        return vocab_list
    
    def bigrams(self):
        # TO DO
        bigram_list = list(self.bigramFreq.items())
        bigram_list.sort(key=lambda x: x[0])      
        bigram_list.sort(key=lambda x: x[1], reverse=True)  

        return bigram_list

    def unigramCount(self, word):
        # TO DO
        return self.unigramFreq.get(word, 0)

    def bigramCount(self, bigram):
        # TO DO
        return self.bigramFreq.get(bigram, 0)

    def unigramProb(self, word):
        # TO DO
        if self.numOfTokens == 0 or self.unigramCount(word) == 0:
            return 0
        
        return self.unigramCount(word) / self.numOfTokens
        # returns unsmoothed unigram probability value

    def bigramProb(self, bigram):
        # TO DO
        word1 = bigram[0]

        bigram_count = self.bigramCount(bigram)
        word1_count = self.unigramCount(word1)

        if word1_count == 0 or bigram_count == 0:
            return 0
        
        return bigram_count / word1_count
        # returns unsmoothed bigram probability value

    def unigramProb_SmoothingUNK(self, word):
        # TO DO
        return (self.unigramCount(word) + 1) / (self.numOfTokens + (self.sizeOfVocab + 1))
        # returns smoothed unigram probability value

    def bigramProb_SmoothingUNK(self, bigram):
        # TO DO
        word1 = bigram[0]

        bigram_count = self.bigramCount(bigram)
        word1_count = self.unigramCount(word1)
        
        return (bigram_count + 1) / (word1_count + (self.sizeOfVocab + 1))
        # returns smoothed bigram probability value

    def sentenceProb(self,sent):
        # TO DO 
        # sent is a list of tokens
        # returns the probability of sent using smoothed bigram probability values

        if len(sent) < 2:
            return self.unigramProb_SmoothingUNK(sent[0])

        s_prob = 1.0

        for i in range(len(sent) - 1):
            bigram = (sent[i], sent[i+1])
            b_prob = self.bigramProb_SmoothingUNK(bigram)
            s_prob *= b_prob
        
        return s_prob

    def generateSentence(self,sent=["<s>"],maxFollowWords=1,maxWordsInSent=20):
        # TO DO 
        # sent is a list of tokens
        # returns the generated sentence (a list of tokens)
        while len(sent) <= maxWordsInSent:
            last_word = sent[-1]  # by last word

            candidates = []
            for (bigram, freq) in self.bigrams():
                if bigram[0] == last_word:
                    prob = self.bigramProb_SmoothingUNK(bigram)
                    candidates.append((bigram[1], prob))
            
             # If there is no candidate finish 
            if not candidates:
                break

            # Sort by probability and choose the maxFollowWords with the highest probability
            candidates.sort(key=lambda x: (-x[1], x[0]))  
            top_candidates = candidates[:maxFollowWords]
            
            words = [w for (w, p) in top_candidates]
            probs = [p for (w, p) in top_candidates]
            
            # Random weighted selection
            next_word = random.choices(words, weights=probs, k=1)[0]

            sent.append(next_word)

            if next_word == "</s>":
                break
            
        if sent[-1] != "</s>":
            sent.append("</s>")
        return sent
    
