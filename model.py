from sklearn.datasets import make_classification
from report import report
from sklearn.svm import SVC
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
from pymagnitude import Magnitude
import numpy as np
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
import tensorflow as tf
import tensorflow_hub as hub
from utility import featurize_and_split, preprocess


glove = Magnitude("vectors/glove.twitter.27B.100d.magnitude")
elmo_vecs = Magnitude("vectors/elmo_2x1024_128_2048cnn_1xhighway_weights.magnitude")
tfidf = TfidfVectorizer()
embed = hub.load('4')






def pipeline_avg_glove(text):
    return np.average(glove.query(word_tokenize(text)), axis = 0)

def avg_glove(X):
    X_train, X_test = X
    train_vectors = []
    test_vectors = []
    for text in tqdm(X_train):
        train_vectors.append(pipeline_avg_glove(text))
    for text in tqdm(X_test):
        test_vectors.append(pipeline_avg_glove(text))
    return (np.array(train_vectors), np.array(test_vectors))


def pipeline_idf_glove(text):
    idf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
    glove_vectors = glove.query(word_tokenize(text))
    weights = [idf_dict.get(word, 1) for word in word_tokenize(text)]
    return np.average(glove_vectors, axis = 0, weights = weights)


def idf_glove(X):
    X_train, X_test = X
    train_vectors = []
    test_vectors = []
    for text in tqdm(X_train): 
        train_vectors.append(pipeline_idf_glove(text))
    for text in tqdm(X_test):
        test_vectors.append(pipeline_idf_glove(text))
    return (np.array(train_vectors), np.array(test_vectors))


def elmo(X):
    X_train, X_test = X
    train_vectors = []
    test_vectors = []
    for text in tqdm(X_train):
        train_vectors.append(elmo_vecs.query(text))
    for text in tqdm(X_test):
        test_vectors.append(elmo_vecs.query(text))
    return (np.array(train_vectors), np.array(test_vectors))


def tfidf_vectorize(X):
    X_train, X_test = X
    X_train_tfidf = tfidf.transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return X_train_tfidf, X_test_tfidf


def read_data(dataset_path):
    print('Reading Data...')
    data = pd.read_csv(dataset_path)
    X, y = data.data.values, data.intent.values
    return X, y

def pipeline_sent_enc(text):
    return embed([text]).numpy()[0]

def sent_enc(X):
    X_train, X_test = X
    train_vectors = embed(X_train).numpy()
    test_vectors = embed(X_test).numpy()
    return train_vectors, test_vectors

if __name__ == '__main__':
    #sentence encoder
    dataset = read_data('dataset/mainModel.csv')
    dataset = preprocess(dataset, {'tfidf' : tfidf})
    features = {
        'sent_enc' : pipeline_sent_enc,
        'glove' : pipeline_avg_glove,
        'idf_glove' : pipeline_idf_glove,
        #'tfidf' : tfidf_vectorize,
        #'elmo' : elmo
    }

    features = featurize_and_split(dataset, features)
    svm = {
        'model' : OneVsRestClassifier(SVC(verbose = True, probability = True)),
        #'params' : {'estimator__C' : [0.1, 1, 10, 50, 100], 'estimator__kernel': ['rbf', 'linear']}
        'params' : None
    }

    svm_sgd = {
        'model' : OneVsRestClassifier(SGDClassifier(loss = 'log'), n_jobs = -1),
        'params' : {'estimator__alpha' : [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
    } 

    rf = {
        'model' : OneVsRestClassifier(RandomForestClassifier()),
        'params' : {'estimator__n_estimators' : [10, 50, 100]}
    }

    xgb = {
        'model' : MultiOutputClassifier(XGBClassifier(), n_jobs= -1),
        'params' : {'estimator__max_depth' : [2,5,7], 'estimator__n_estimators': [100]}
    }
    models = {
       'svm' : svm_sgd,
       #'rf' : rf
    }


    report(models, features)
  
