#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 17:33:28 2018

@author: neha
"""
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from nltk.corpus import gutenberg
import numpy as np
from string import digits
import regex as re
import sys

def load():
    test1=[]    
    train1=[]
    train=[]
    validation=[]
    test=[]  

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
            str3= str3 + i.translate(str.maketrans('', '', digits)).lower()
#            str3= str3+ ' <s> '+ i.translate(str.maketrans('','',string.punctuation)).translate(str.maketrans('', '', digits))
                
        str3=re.sub("[^\P{P}]+", "", str3)

#        punctuation={'`','\''}
#        for c in punctuation:
#            str3= str3.replace(c,"")
            
        punctuation={' s ',' d ',' t ',' ve ',' ll ',' \'', ' st ', ' nd ', ' rd ', '`' , '$', '>'}
        for c in punctuation:
            str3= str3.replace(c,"")
#        
#        str3= str3.replace(" - "," - ".strip())
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
    
    test = test[:round(len(test)*0.2)]
    train = train[:round(len(train)*0.2)]
    
    raw_text=' '
    raw_text=' '.join(train)

    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    
    n_chars = len(raw_text)
    n_vocab = len(chars)
#    print ("Total Characters: ", n_chars)
#    print ("Total Vocab: ", n_vocab)
    
    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 120
    dataX = []
    dataY = []
    
    for i in range(0, n_chars - seq_length, 1):
        	seq_in = raw_text[i:i + seq_length]
        	seq_out = raw_text[i + seq_length]
        	dataX.append([char_to_int[char] for char in seq_in])
        	dataY.append(char_to_int[seq_out])
    
    n_patterns = len(dataX)
#    print ("Total Patterns: ", n_patterns)

    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)
    
    
    
    raw_text=' '
    raw_text=' '.join(test)

#    chars = sorted(list(set(raw_text)))
#    char_to_int = dict((c, i) for i, c in enumerate(chars))
#    int_to_char = dict((i, c) for i, c in enumerate(chars))
    
    n_chars = len(raw_text)
    n_vocab = len(chars)
#    print ("Total Characters: ", n_chars)
#    print ("Total Vocab: ", n_vocab)
    
    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 120
    dataX_test = []
    dataY_test = []
    
    for i in range(0, n_chars - seq_length, 1):
        	seq_in = raw_text[i:i + seq_length]
        	seq_out = raw_text[i + seq_length]
        	dataX_test.append([char_to_int[char] for char in seq_in])
        	dataY_test.append(char_to_int[seq_out])
    
    n_patterns = len(dataX_test)
    
    # reshape X to be [samples, time steps, features]
    X_test = numpy.reshape(dataX_test, (n_patterns, seq_length, 1))
    # normalize
    X_test = X_test / float(n_vocab)
    # one hot encode the output variable
    y_test = np_utils.to_categorical(dataY_test)
    
    return X,y,train,test,n_vocab,int_to_char,dataX_test,X_test,y_test

def fit_model(X,y):
    # define the LSTM model
    model = Sequential()
    model.add(LSTM(200, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    
    # define the checkpoint
    filepath="weights1-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
    
    
def genSent(X,y,n_vocab,int_to_char,dataX):    
    
    model = Sequential()
    model.add(LSTM(200, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # load the network weights
    filename = "weights-improvement-19-2.8582.hdf5"
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # pick a random seed
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    # generate characters
    for i in range(50):
        	x = numpy.reshape(pattern, (1, len(pattern), 1))
        	x = x / float(n_vocab)
        	prediction = model.predict(x, verbose=0)
        	index = numpy.argmax(prediction)
        	result = int_to_char[index]
        	seq_in = [int_to_char[value] for value in pattern]
        	sys.stdout.write(result)
        	pattern.append(index)
        	pattern = pattern[1:len(pattern)]

def evaluate_model(X_test,y_test):
    model = Sequential()
    model.add(LSTM(200, input_shape=(X_test.shape[1], X_test.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y_test.shape[1], activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # load the network weights
    filename = "weights-improvement-19-2.8582.hdf5"
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    score= model.evaluate(X_test, y_test,batch_size=128)
    print("Score:",score)
    
    
X,y,train,test,n_vocab,int_to_char,dataX_test,X_test,y_test=load()
fit_model(X,y)
evaluate_model(X_test,y_test)
#genSent(X,y,n_vocab,int_to_char,dataX_test)
