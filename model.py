from sklearn.datasets import make_classification
from report import report
from utility import featurize_and_split, preprocess
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


def avg_glove(X):
    X_train, X_test = X
    glove = Magnitude("vectors/glove.twitter.27B.100d.magnitude")
    train_vectors = []
    test_vectors = []
    for text in tqdm(X_train):
        train_vectors.append(np.average(glove.query(word_tokenize(text)), axis = 0))
    for text in tqdm(X_test):
        test_vectors.append(np.average(glove.query(word_tokenize(text)), axis = 0))
    return (np.array(train_vectors), np.array(test_vectors))


def get_idf_glove(X, glove, idf_dict):
    vectors = []
    for text in tqdm(X):
        glove_vectors = glove.query(word_tokenize(text))
        weights = [idf_dict.get(word, 1) for word in word_tokenize(text)]
        vectors.append(np.average(glove_vectors, axis = 0, weights = weights))
    return np.array(vectors)


def idf_glove(X):
    X_train, X_test = X
    tfidf = TfidfVectorizer()
    tfidf.fit(X_train)
    idf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
    glove = Magnitude("vectors/glove.twitter.27B.100d.magnitude")
    train_vectors = get_idf_glove(X_train, glove, idf_dict)
    test_vectors = get_idf_glove(X_test, glove, idf_dict)
    return train_vectors, test_vectors


def elmo(X):
    X_train, X_test = X
    elmo_vecs = Magnitude("vectors/elmo_2x1024_128_2048cnn_1xhighway_weights.magnitude")

    train_vectors = []
    test_vectors = []
    for text in tqdm(X_train):
        train_vectors.append(elmo_vecs.query(text))
    for text in tqdm(X_test):
        test_vectors.append(elmo_vecs.query(text))
    return (np.array(train_vectors), np.array(test_vectors))


def tfidf_vectorize(X):
    X_train, X_test = X
    tfidf = TfidfVectorizer(max_features = 300)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return X_train_tfidf, X_test_tfidf


def read_data(dataset_path):
    print('Reading Data...')
    data = pd.read_csv(dataset_path)
    X, y = data.data.values, data.intent.values
    return X, y


if __name__ == '__main__':
    #sentence encoder
    dataset = read_data('dataset/mainModel.csv')
    dataset = preprocess(dataset)
    features = {
        'glove' : idf_glove,
        #'tfidf' : tfidf_vectorize,
        'elmo' : elmo
    }

    features = featurize_and_split(dataset, features)
    svm = {
        'model' : OneVsRestClassifier(SVC(verbose = True, probability = True)),
        'params' : {'estimator__C' : [0.1, 1, 10, 50, 100], 'estimator__kernel': ['rbf', 'linear']}
    }

    svm_sgd = {
        'model' : OneVsRestClassifier(SGDClassifier(loss = 'log')),
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
       'rf' : rf
    }


    report(models, features)
  
