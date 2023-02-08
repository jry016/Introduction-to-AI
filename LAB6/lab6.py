# Remember to include the import line(s) if you have use valid module(s) other than the one listed here
import pandas as pd
import numpy as np
import re
import keras
import os

from numpy import array
from keras.preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from tensorflow.keras import regularizers
from keras.layers import Flatten
from keras.layers import Embedding,BatchNormalization
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.initializers import HeNormal
import matplotlib.pyplot as plt


def preprocessing(sentence):
    # remove html tags
    tag = re.compile(r'<[^>]+>')
    sentence = tag.sub('', sentence)
    # remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # remove single char
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


def get_X_y(df):

  return X,y


def readglovefile(filepath):
  import gzip
  with gzip.open(filepath,'r') as f:
    content=f.readlines()
  return [i.decode('utf8') for i in content]


def formatEmbDict(filepath):
  wordDict = {}
  contents = readglovefile(filepath)

  return wordDict


def myModel(vocab_size, embedding_matrix, maxlen):
  model = Sequential()
  # in embedding_layer, the word index array for each instance is transformed to the GloVe embedding according to the embeddings matrix
  embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
  model.add(embedding_layer)


  return model
