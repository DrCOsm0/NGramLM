import pickle, random
from nltk import word_tokenize
from nltk.util import ngrams

def predict_next(prompt):
    
    with open('gen_ngram/ngrams_prob.pkl', 'rb') as file:
        ngrams_prob = pickle.load(file)

    str1 = prompt
    token_1 = word_tokenize(str1)

    ngram_1 = {1:[], 2:[], 3:[]}

    for i in range(3): #change for diff num of words?
        ngram_1[i+1] = list(ngrams(token_1, i+1))[-1]

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

    return pred_1[1] #return bigrams only

def predict(prompt, num_words):
    
    i = 0
    while(i < num_words):
        words = predict_next(prompt)
        next_word = random.choice(words)
        if next_word == 'NOT FOUND':#add if fourgram not found, use tri gram, else use bigram
            while next_word == 'NOT FOUND':
                next_word = random.choice(words)
        prompt = prompt + " " + next_word
        i = i + 1
    
    return prompt

output = predict("alice felt so desperate that", 10)
print(output) 
