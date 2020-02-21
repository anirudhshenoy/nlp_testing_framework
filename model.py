from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from report import Model, report
from sklearn.svm import SVC
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm
from pymagnitude import Magnitude
import numpy as np
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier



def split_dataset(X, y, split_type = 'split', filepath = ''):
    if split_type == 'split':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify = y)
    else:
        (X_test, y_test) = read_test_file(filepath)
        (X_train, y_train) = X, y
    return (X_train, y_train), (X_test, y_test)

def read_test_file(filepath):
    test_data = pd.read_csv(filepath)
    test_data = test_data[test_data['Intent'].isin(['apply-leave', 'leave-balance', 'schedule-meeting'])]
    test_data.columns = ['intent', 'data']
    (X_test, y_test) = (test_data.data, test_data.intent)
    return (X_test, y_test)


def featurize_and_split(dataset, features):
    print('Featurizing...')

    X, y = dataset
    (X_train, y_train), (X_test, y_test) = split_dataset(X, y, 
                    split_type = 'file', filepath = 'dataset/training-set.csv')

    le = LabelBinarizer()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    for feature_name, feature in features.items():
        #X_train, X_test = tfidf_vectorize((X_train, X_test))
        #X_train, X_test = avg_glove((X_train, X_test))
        X_train_feat, X_test_feat = feature((X_train, X_test))
        features[feature_name] = {
            'featurizer' : feature,
            'feature_data' : ((X_train_feat, y_train), (X_test_feat, y_test))
        }
    print('glove done')
    return features 
    #return (X_train, y_train), (X_test, y_test)

def avg_glove(X):
    X_train, X_test = X
    glove = Magnitude("vectors/glove.6B.50d.magnitude")
    train_vectors = []
    test_vectors = []
    for text in tqdm(X_train):
        train_vectors.append(np.average(glove.query(word_tokenize(text)), axis = 0))
    for text in tqdm(X_test):
        test_vectors.append(np.average(glove.query(word_tokenize(text)), axis = 0))
    return (train_vectors, test_vectors)

def tfidf_vectorize(X):
    X_train, X_test = X
    tfidf = TfidfVectorizer(max_features = 300)
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)
    return X_train, X_test

def preprocess(dataset):
    print('Preprocessing...')

    # Preprocessing pipeline
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   
    X_test = scaler.transform(X_test)
    return (X_train, y_train), (X_test, y_test)
    """
    return dataset

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
  
