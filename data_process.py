import os
import operator
import string
import pickle

import numpy as np
import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit

def tokenize(text):
    tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return [f.lower() for f in tokens if f and f.lower() not in stop_words]

def get20News():
    dir = './20_newsgroups'
    category_names = []
    for f in os.listdir(dir):
        if not f.startswith('.'):
            category_names.append(f)
    news_contents = list()
    news_labels = list()
    for i in range(len(category_names)):
        category = category_names[i]
        category_dir = os.path.join(dir, category)
        for file_name in os.listdir(category_dir):
            file_path = os.path.join(category_dir, file_name)
            raw_content = open(file_path, encoding='latin1').read().strip()
            news_content = tokenize(raw_content)
            news_labels.append(i)
            news_contents.append(news_content)
    return news_contents, np.array(news_labels), category_names

def getEmbeddingMatrix(word_index, vector):
    wordvector = {'fasttext':'./vectors/crawl-300d-2M.vec'}
    f = open(wordvector[vector])
    allWv = {}
    if vector=='fasttext':
        errcnt = 0
        for line in f:
            values = line.split()
            word = values[0].strip()
            try:
                wv = np.asarray(values[1:], dtype='float32')
                if len(wv)!=300:
                    errcnt+=1
                    continue
            except:
                errcnt+=1
                continue
            allWv[word] = wv
        print('# bad word vector num: ', errcnt)
    f.close()
    embedding_matrix = np.zeros((len(word_index)+1, 300))
    for word, i in word_index.items():
        if word in allWv:
            embedding_matrix[i] = allWv[word]
    return embedding_matrix

def sparseMultiply(X, embedding_matrix):
    denseZ = []
    for row in X:
        newrow = np.zeros(300)
        for nonzeropos, value in list(zip(row.indices, row.data)):
            newrow += value*embedding_matrix[nonzeropos]
        denseZ.append(newrow)
    denseZ = np.array([np.array(x) for x in denseZ])
    return denseZ

def generate_file():
    vector = 'fasttext'
    print('-----------------start load data-----------------')
    X, labels, category_names = get20News()
    X = np.array([np.array(x) for x in X])
    print('-----------------load data finished-----------------')

    print('-----------------start TF-IDF-----------------')
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1).fit(X)
    word_index = vectorizer.vocabulary_
    X_encoded = vectorizer.transform(X)
    print('-----------------TF-IDF finished-----------------')

    text_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(X_encoded, labels)
    train_indices, test_indices = next(text_split)

    x_train, x_test = X_encoded[train_indices], X_encoded[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]

    with open('./data_tfidf/x_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)
    with open('./data_tfidf/x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)
    with open('./data_tfidf/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('./data_tfidf/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)

    print('-----------------start embedding-----------------')
    embedding_matrix = getEmbeddingMatrix(word_index, vector)
    X_encoded = sparseMultiply(X_encoded, embedding_matrix)
    print('-----------------embedding finished-----------------')

    text_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(X_encoded, labels)
    train_indices, test_indices = next(text_split)

    x_train, x_test = X_encoded[train_indices], X_encoded[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    with open('./data/x_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)
    with open('./data/x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)
    with open('./data/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('./data/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)

if __name__=='__main__':
    generate_file()