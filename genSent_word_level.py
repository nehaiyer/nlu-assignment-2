#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:36:14 2018

@author: neha
"""
import numpy as np
import nltk
from nltk.corpus import gutenberg
from string import digits
import regex as re
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def load_train():
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

#    print("done1")
    train = [item for sublist in train1 for item in sublist]
    test = [item for sublist in test1 for item in sublist]
    
    
    data = train + test
    word_freq = nltk.FreqDist(data)
#    print ("Found %d unique words tokens.", len(word_freq.items()))
    
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    
#    print ("Using vocabulary size %d.", vocabulary_size)
#    print ("The least frequent word in our vocabulary is '%s' and appeared %d times.",(vocab[-1][0], vocab[-1][1]))
    
    for i, w in enumerate(train):
        if w in word_to_index:
            train[i] = w
        else:
            train[i] = unknown_token
#            print("UKN")

    for i, w in enumerate(test):
        if w in word_to_index:
            test[i] = w
        else:
            test[i] = unknown_token
#            print("UKN")
    
    
#    validation = train[-round(len(train)*0.8):]
    train = train[:round(len(train)*0.2)]
    test = test[:round(len(test)*0.2)]

    return train,test

def generate_sequences(train):
    length = 50 + 1
    sequences = []
    for i in range(length, len(train)):
        	# select sequence of tokens
        	seq = train[i-length:i]
        	# convert into a line
#        	if len(seq) != 51:
#        	    	print("length ",len(seq))
        	line = ' '.join(seq)
        	# store
        	sequences.append(line)
#    print('Total Sequences: %d' % len(sequences))
    
    return sequences

def batch_producer(sequences):
    tokenizer = Tokenizer(split=' ',filters='')
    tokenizer.fit_on_texts(sequences)
    sequences1 = tokenizer.texts_to_sequences(sequences)
    
    vocab_size = len(tokenizer.word_index) + 1
    sequences2 = np.array(sequences1,np.int32)
    X, y = sequences2[:,:-1], sequences2[:,-1]
    seq_length = X.shape[1]
#    y = to_categorical(y_new, num_classes=vocab_size)
    return X,y,vocab_size,seq_length,tokenizer

def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)

train,test=load_train()
sequences=generate_sequences(train)
X,y,vocab_size,seq_length,tokenizer = batch_producer(sequences)

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
#print(model.summary())
# compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model = load_model('/home/neha/sem2/nlu/ass2/model_new_final1.h5')
 # load the tokenizer
tokenizer1 = load(open('/home/neha/sem2/nlu/ass2/tokenizer.pkl', 'rb'))
# select a seed text

sequences=generate_sequences(test)
seed_text = sequences[randint(0,len(sequences))]
print("Seed text: ", seed_text + '\n')

generated = generate_seq(model, tokenizer1, seq_length, seed_text, 11)
print("Generated text: ",generated + '\n')