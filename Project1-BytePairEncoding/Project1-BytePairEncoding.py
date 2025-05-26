"""
Assignment 1 - Code template for AIN442/BBM497

"""
import re
import codecs

def initialVocabulary():
    
    # You can use this function to create the initial vocabulary.
    
    return list("abcçdefgğhıijklmnoöprsştuüvyzwxq"+
                "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ"+
                "0123456789"+" "+
                "!'^#+$%&/{([)]=}*?\\_-<>|.:´,;`@€¨~\"é")

def bpeCorpus(corpus, maxMergeCount=10):     

    # TO DO
    # You can refer to Example 1, 2 and 3 for more details.
    
    words = re.split(r'\s+', corpus)

    # adding " " to the beginning and _ to the end
    Tokenized_corpus = []

    for word in words:
        tokenized_word = [" "] + list(word) + ["_"]
        Tokenized_corpus.append(tokenized_word)

    Merges = []
    
    for _ in range(maxMergeCount):
        # counting bigrams 
        bigram_freq = {}

        for t_corp in Tokenized_corpus:        
            for i in range(len(t_corp) - 1):
                bigram = (t_corp[i] , t_corp[i+1])
                if bigram in bigram_freq:
                    bigram_freq[bigram] += 1
                else:
                    bigram_freq[bigram] = 1

        if not bigram_freq:
            break  

        # merging process

        best_bigram = None
        max_count = 0

        for bigram, count in bigram_freq.items():
            if count > max_count or (count == max_count and bigram < best_bigram):
                best_bigram = bigram
                max_count = count

        if best_bigram is None:
            break

        Merges.append((best_bigram, max_count))

        # creating new tokenized corpus
        for t_corp in Tokenized_corpus:
            i = 0
            while i < len(t_corp) - 1:  
                bigram = (t_corp[i], t_corp[i+1])
                if bigram == best_bigram:
                    merged_corp = t_corp[i] + t_corp[i+1]  
                    t_corp[i] = merged_corp  
                    t_corp.pop(i+1)  
                else:
                    i += 1  
        
    Vocabulary = initialVocabulary() +  [merge[0][0] + merge[0][1] for merge in Merges]


    return (Merges, Vocabulary, Tokenized_corpus) # Should return (Merges, Vocabulary, TokenizedCorpus)


def bpeFN(fileName, maxMergeCount=10):

    # TO DO
    # You can refer to Example 4 and 5 for more details.

    with codecs.open(fileName, "r", encoding="utf-8") as f:
        corpus = f.read()

    # Clearing blank lines 
    lines = corpus.split("\n")
    corpus = " ".join([line.strip() for line in lines if line.strip()])  
    

    return bpeCorpus(corpus, maxMergeCount) # Should return (Merges, Vocabulary, TokenizedCorpus)

def bpeTokenize(str, merges):

    # TO DO
    # You can refer to Example 6, 7 and 8 for more details.

    words = re.split(r'\s+', str)

    # adding " " to the beginning and _ to the end
    Tokenized_corpus = []

    for word in words:
        tokenized_word = [" "] + list(word) + ["_"]
        Tokenized_corpus.append(tokenized_word)
    
    for merge_bigram, _ in merges:
        for t_corp in Tokenized_corpus:
                i = 0
                while i < len(t_corp) - 1:  
                    bigram = (t_corp[i], t_corp[i+1])
                    if bigram == merge_bigram:
                        merged_corp = t_corp[i] + t_corp[i+1]  
                        t_corp[i] = merged_corp 
                        t_corp.pop(i+1)  
                    else:
                        i += 1  

    return Tokenized_corpus # Should return the tokenized string as a list


def bpeFNToFile(infn, maxMergeCount=10, outfn="output.txt"):
    
    # Please don't change this function. 
    # After completing all the functions above, call this function with the sample input "hw01_bilgisayar.txt".
    # The content of your output files must match the sample outputs exactly.
    # You can refer to "Example Output Files" section in the assignment document for more details.
    
    (Merges,Vocabulary,TokenizedCorpus)=bpeFN(infn, maxMergeCount)
    outfile = open(outfn,"w",encoding='utf-8')
    outfile.write("Merges:\n")
    outfile.write(str(Merges))
    outfile.write("\n\nVocabulary:\n")
    outfile.write(str(Vocabulary))
    outfile.write("\n\nTokenizedCorpus:\n")
    outfile.write(str(TokenizedCorpus))
    outfile.close()

bpeFNToFile("hw01_bilgisayar.txt", 1000, "hw01-output1.txt")
bpeFNToFile("hw01_bilgisayar.txt", 200, "hw02-output2.txt")
