#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:05:12 2018

@author: neha
"""
import nltk
from nltk.corpus import gutenberg
import numpy as np
import random
from string import digits
import regex as re

def load():
    data=[]
    test1=[]    
    train1=[]
    train=[]
    validation=[]
    test=[]  
    vocabulary_size = 40000
    unknown_token = "UNKNOWN_TOKEN"
    for fileid in gutenberg.fileids():
        sent=gutenberg.sents(fileid)
#        sent=gutenberg.words(fileid)
        s=[]
        for str1 in sent:
            s.append(str1)

        
        str2=[]
        for i in s:
            str2.append(' '.join(i))
        
        str3=''
        for i in str2:
            str3= str3+ " "+i.translate(str.maketrans('', '', digits)).lower()
#            str3= str3+ ' <s> '+ i.translate(str.maketrans('','',string.punctuation)).translate(str.maketrans('', '', digits))
                
        str3=re.sub("[^\P{P}]+", "", str3)

#        punctuation={'`','\''}
#        for c in punctuation:
#            str3= str3.replace(c,"")
            
        punctuation={' s ',' d ',' t ',' ve ',' ll ',' \'', ' st ', ' nd ', ' rd '}
        for c in punctuation:
            str3= str3.replace(c,"")
#        
        str3= str3.replace(" - "," - ".strip())
#        punctuation={'-',' , ',' ? ', ' } ', ' [ ', ' ] ', ' ! ', ' @   '}
#        for c in punctuation:
#            str3= str3.replace(c,c.lstrip())
 
        str3=' '.join(str3.split())
    #    str3=str3.translate(str.maketrans('','',string.punctuation))
    #    str3 = '<s> The Fulton County Grand Jury said Friday an investigation of Atlantas recent primary election produced no evidence that any irregularities took place . <s> The jury further said in term-end presentments that the City Executive Committee , which had over-all charge of the election , deserves the praise and thanks of the City of Atlanta for the manner in which the election was conducted . <s> The September-October term jury had been charged by Fulton Superior Court Judge Durwood Pye to investigate reports of possible irregularities in the hard-fought primary which was won by Mayor-nominate Ivan Allen Jr. .'
        words = str3.split(' ')
        train1.append(words[:round(len(words)*0.8)])
        test1.append(words[-round(len(words)*0.2):])

    print("done1")
    train = [item for sublist in train1 for item in sublist]
    test = [item for sublist in test1 for item in sublist]
    
    
    data = train + test
    word_freq = nltk.FreqDist(data)
    print ("Found %d unique words tokens.", len(word_freq.items()))
    
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    
    print ("Using vocabulary size %d.", vocabulary_size)
    print ("The least frequent word in our vocabulary is '%s' and appeared %d times.",(vocab[-1][0], vocab[-1][1]))
    
    for i, w in enumerate(train):
        if w in word_to_index:
            train[i] = w
        else:
            train[i] = unknown_token
            print("UKN")

    for i, w in enumerate(test):
        if w in word_to_index:
            test[i] = w
        else:
            test[i] = unknown_token
            print("UKN")
    
    
    validation = train[-round(len(train)*0.8):]
    train = train[:round(len(train)*0.2)]
    test = test[:round(len(test)*0.2)]

    return train,test

def cal_ngram(train,n):
    ngrams = {} 
    #n=2
    for index, word in enumerate(train):
        if index < len(train)-(n-1):
            w=[]
            for i in range(n):
                w.append(train[index+i])
            ngram = tuple(w)
#            print(ngram)
    
            if ngram in ngrams:
                ngrams[ ngram ] = ngrams[ ngram ] + 1
            else:
                ngrams[ ngram ] = 1
                
#    sorted_ngrams = sorted(ngrams.items(), key = lambda pair:pair[1], reverse = True)
    return ngrams


def cal_ngram_list(ngrams):
    ngrams_list=[]
    for key,value in ngrams.items():
        ngrams_list.append(key)
    
    return ngrams_list

def unknown(unigrams,train):
    unknown_list=[]
    for key, value in unigrams.items():
        if value < 2:
            unknown_list.append(key[0])
            for index, word in enumerate(unigrams):
                if train[index] == key[0]:
                    train[index] = '<UKN>'
        if len(unknown_list)==500:
                    break
    return train,unknown_list


def cal_probab(ngrams,n_1grams,n):
    prob = {}
    for key, value in ngrams.items():
        n_1key=[]
        for k in range(0,n-1):
            n_1key.append(key[k])
        
        prob[key] = value/(n_1grams[tuple(n_1key)])
        
    return prob


def cal_unigram_probab(ngrams,N):
    prob = {}
    for key, value in ngrams.items():
        prob[key] = value/N 
    return prob


def check_existence(key,ngram_list,train_prob,n):
    found=0
    nfound=0
    alpha=1;
    t_prob=-1
    for i in reversed(range(len(ngram_list))):
#        print(i)
#        print(key)
#        k=[]
#        k.append(key)
        if key in ngram_list[i]:
            prob = train_prob[i]
            t_prob = alpha*prob[key]
#            print('break')
            found=found+1
            break
        else:
            key=key[i:n]
#            print(key)
            alpha=alpha*0.4
            
    if t_prob==-1:
#        print('unknown')
        ukn=tuple(['<UKN>'])
        nfound=nfound+1
        prob = train_prob[0]
        t_prob=alpha*prob[ukn]/0.4
    return t_prob
                
def cal_probab_test(tngram,ngram_list,train_prob,n):
    t_prob=0
    
    for key, value in tngram.items():
#        print(key)
        prob = check_existence(key,ngram_list,train_prob,n)
#        print(prob)
        t_prob = t_prob + np.log2(prob)
        
    return t_prob

def cal_perplexity(test,ngram_list,train_prob,n):
    tngram={}
    tngram=cal_ngram(test,n)
    tN=len(test)
    tprob=cal_probab_test(tngram,ngram_list,train_prob,n)
    perplexity=2 ** (tprob*(-1/tN))
    
    return perplexity

def init(train,n):
    N=len(train)
    unigrams=cal_ngram(train,1)
    #replace some vocab with <UKN>
    train,unknown_list = unknown(unigrams,train)
    
    #get all ngrams and their counts
    ngram=[]
    
    for i in range(n):
#        print(i)
        ngram.append(cal_ngram(train,i+1))
    
    #calculate 1 to n gram's probabilities
    train_prob=[]
    train_prob.append(cal_unigram_probab(ngram[0],N))
    
    for i in range(1,n):
#        print(i)
        train_prob.append(cal_probab(ngram[i],ngram[i-1],i+1))
    
    
    
    #calculate ngram lists
    ngram_list=[]
    for i in range(n):
        ngram_list.append(cal_ngram_list(ngram[i]))
        
    return N,n,train,unknown_list,ngram,train_prob,ngram_list

def generate_sent(word, length, ngram,n,unknown_list):
    sent=word
    keylist=[]
    wordlist=[]
    newsent=''
    keylist.append(word)
    wordlist.append(word)
    word=tuple(keylist)
    for i in range(1,n-1):
#        print(i)
        word=get_next_word(word,ngram[i],i,n,unknown_list)
        if word != '<s>':
            wordlist.append(word)
        if word == '<s>':
            i=i-1
#        print(word)
        keylist.append(word)
        word=tuple(keylist)
#    sent = sent + ' '+ word
#    key=tuple(keylist)
#    
#    print(word)
    for i in range(n-1,length+1):
#        print(i)
        word=get_next_word(word,ngram[n-1],n-1,n,unknown_list)
        if word != '<s>':
            wordlist.append(word)
        if word == '<s>':
            i=i-1
        keylist.append(word)
        word=tuple(keylist)
        word=word[len(keylist)-(n-1):len(keylist)]
#        print(word)
        
#    newsent = ' '.join(wordlist)
    word=tuple(wordlist)
#        sent = sent + ' '+ word
    return word
        

def get_next_word(word,ngram,i,n,unknown_list):
    
    maxcount=0
    if i < n-1:
        for key,value in ngram.items():
            if tuple(list(key[0:i])) == word:
                if maxcount < value:
                    maxcount=value
                    maxkey=key[i]
    
    else:
        for key,value in ngram.items():
#            print(tuple(list(key[0:i])))
#            print(word)
            if tuple(list(key[0:i])) == word:
                
                if maxcount < value:
                    maxcount=value
                    maxkey=key[i]
                    
    if maxcount==0:
        maxkey=random.choice(unknown_list)

    return maxkey

train,test=load()
n=2
N,n,train,unknown_list,ngram,train_prob,ngram_list=init(train,n)

unigram=ngram[0]
sorted_unigrams = dict(sorted(unigram.items(), key = lambda pair:pair[1], reverse = True))

unigram_list=[]


for index, word in enumerate(sorted_unigrams):
    unigram_list.append(word[0])
    if len(unigram_list) == 100:
        break
    

startword= random.choice(unigram_list[0:500])
gensent=generate_sent(startword, 10, ngram,n,unknown_list)
gensent=list(gensent)
gensent = ' '.join(gensent)
print(gensent)