from sklearn.datasets import make_classification
from report import report
from utility import featurize_and_split, preprocess
from sklearn.svm import SVC
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm
from pymagnitude import Magnitude
import numpy as np
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier


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
    
    dataset = read_data('dataset/mainModel.csv')
    dataset = preprocess(dataset)
    features = {
        'glove' : avg_glove,
        'tfidf' : tfidf_vectorize,
    }

    features = featurize_and_split(dataset, features)
    svm = {
        'model' : MultiOutputClassifier(SVC(probability = True)),
        'params' : {'estimator__C' : [0.1, 1, 10, 50, 100], 'estimator__kernel': ['rbf', 'linear']}
    }

    rf = {
        'model' : MultiOutputClassifier(RandomForestClassifier()),
        'params' : {'estimator__n_estimators' : [10, 50, 100]}
    }

    models = {
        'svm' : svm,
        'rf' : rf
    }


    report(models, features)
  
