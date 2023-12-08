import nltk, re, string
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords

# Add Punctuation 
string.punctuation = string.punctuation + '“' + '”' +'-' + '’' + '‘' + '—'
string.punctuation = string.punctuation.replace('.', '')

# Read File
file = open('corpus.txt', encoding='utf8').read()

# Remove all new lines
file_nl_removed = ""
for line in file:
    line_nl_removed = line.replace("\n", " ")
    file_nl_removed += line_nl_removed

#Removes all punctuation except periods
file_p = "".join([char for char in file_nl_removed if char not in string.punctuation])
sents = nltk.sent_tokenize(file_p) #uses . to tokenize all sentences into a list (where each element is a sentence)

# Input text stats
print("The number of sentences is", len(sents))
words = nltk.word_tokenize(file_p)
print("The number of tokens is", len(words))
average_tokens = round(len(words)/len(sents))
print("The average number of words per sentence is", average_tokens)
unique_tokens = set(words)
print("The number of unique words are", len(unique_tokens))

stop_words = set(stopwords.words('english'))
unigram=[]
bigram=[]
trigram=[]
fourgram=[]
tokenized_text = []

# tokenized_text = List of Lists (where each sublist is a sentence) and (each element of the sublist is a word)
# EXAMPLE : [['this', 'is', 'sentence', 'one'],['this', 'is', 'sentence', 'two']]

for sentence in sents:
    sentence = sentence.lower()
    tokenized_sentence = word_tokenize(sentence)
    #Check for periods and remove
    for word in tokenized_sentence:
        if (word =='.'):
            tokenized_sentence.remove(word) 
        else:
            unigram.append(word)
    tokenized_text.append(tokenized_sentence)
    # Split into correspoding ngram sublists 
    bigram.extend(list(ngrams(tokenized_sentence, 2)))              
    trigram.extend(list(ngrams(tokenized_sentence, 3)))
    fourgram.extend(list(ngrams(tokenized_sentence, 4)))
    
#remove ngrams which only contain stop words
def removal(x):
    y = []
    for pair in x:
        count = 0
        for word in pair:
            if word in stop_words:
                count = count or 0
            else:
                count = count or 1
        if (count==1):
            y.append(pair)
    return(y)

# Each N-Gram is a list of tuples
bigram = removal(bigram)
trigram = removal(trigram)             
fourgram = removal(fourgram)

freq_bi = nltk.FreqDist(bigram)
freq_tri = nltk.FreqDist(trigram)
freq_four = nltk.FreqDist(fourgram)

# N-Gram Stats without add-1 smoothing and partial stopword removal
print("Most common n-grams partial stopword removal and without add-1 smoothing: \n")
print ("Most common bigrams: ", freq_bi.most_common(10))      
print ("\nMost common trigrams: ", freq_tri.most_common(10))
print ("\nMost common fourgrams: ", freq_four.most_common(10))

print(tokenized_text)
exit()

# Add-1 Smoothing
ngrams_all = {1:[], 2:[], 3:[], 4:[]}
for i in range(4):
    for each in tokenized_text:
        for j in ngrams(each, i+1):
            ngrams_all[i+1].append(j)

ngrams_voc = {1:set([]), 2:set([]), 3:set([]), 4:set([])}

for i in range(4):
    for gram in ngrams_all[i+1]:
        if gram not in ngrams_voc[i+1]:
            ngrams_voc[i+1].add(gram)

total_ngrams = {1:-1, 2:-1, 3:-1, 4:-1}
total_voc = {1:-1, 2:-1, 3:-1, 4:-1}
for i in range(4):
    total_ngrams[i+1] = len(ngrams_all[i+1])
    total_voc[i+1] = len(ngrams_voc[i+1])                       

ngrams_prob = {1:[], 2:[], 3:[], 4:[]}
for i in range(4):
    for ngram in ngrams_voc[i+1]:
        tlist = [ngram]
        tlist.append(ngrams_all[i+1].count(ngram))
        ngrams_prob[i+1].append(tlist)

for i in range(4):
    for ngram in ngrams_prob[i+1]:
        ngram[-1] = (ngram[-1]+1)/(total_ngrams[i+1] + total_voc[i+1])             

print("Most common n-grams without stopword removal and with add-1 smoothing: \n")
for i in range(4):
    ngrams_prob[i+1] = sorted(ngrams_prob[i+1], key = lambda x:x[1], reverse = True)

print ("Most common unigrams: ", str(ngrams_prob[1][:10]))
print ("\nMost common bigrams: ", str(ngrams_prob[2][:10]))
print ("\nMost common trigrams: ", str(ngrams_prob[3][:10]))
print ("\nMost common fourgrams: ", str(ngrams_prob[4][:10]))

# Predict 

str1 = 'after that alice said the'
str2 = 'alice felt so desperate that she was'

input_token_1 = word_tokenize(str1)
input_token_2 = word_tokenize(str2)

ngram_1 = {1:[], 2:[], 3:[]}
ngram_2 = {1:[], 2:[], 3:[]}
for i in range(3):
    ngram_1[i+1] = list(ngrams(input_token_1, i+1))[-1]
    ngram_2[i+1] = list(ngrams(input_token_2, i+1))[-1]

print("String 1: ", ngram_1,"\nString 2: ",ngram_2)

pred_1 = {1:[], 2:[], 3:[]}
for i in range(3):
    count = 0
    for each in ngrams_prob[i+2]:
        if each[0][:-1] == ngram_1[i+1]:                      
            count +=1
            pred_1[i+1].append(each[0][-1])
            if count ==5:
                break
    if count<5:
        while(count!=5):
            pred_1[i+1].append("NOT FOUND")           
            count +=1
