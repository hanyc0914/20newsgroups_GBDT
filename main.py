from data_process import *
from gbdt_model import gbdt_model, svm, gbdt_model_grid1, gbdt_model_default
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

if __name__=='__main__':
    model = ['fasttext', 'tfidf', 'tfidf_gridsearch', 'tfidf_best']
    # default choose tfidf with best params
    method = 'tfidf_best'

    if method is model[0]:
        with open('./data/x_train.pkl', 'rb') as f:
            x_train = pickle.load(f)
        with open('./data/x_test.pkl', 'rb') as f:
            x_test = pickle.load(f)
        with open('./data/y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open('./data/y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)
        gbdt_model_default(x_train, y_train, x_test, y_test)
    else:
        with open('./data_tfidf/x_train.pkl', 'rb') as f:
            x_train = pickle.load(f)
        with open('./data_tfidf/x_test.pkl', 'rb') as f:
            x_test = pickle.load(f)
        with open('./data_tfidf/y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open('./data_tfidf/y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)

        if method is model[1]:
            gbdt_model_default(x_train, y_train, x_test, y_test)
        elif method is model[2]:
            gbdt_model_grid1(x_train, y_train, x_test, y_test)
        elif method is model[3]:
            gbdt_model(x_train, y_train, x_test, y_test)