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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lstm import LSTM_Model
from dnn import DNN_Model
from cnn import CNN_Model


glove = Magnitude("vectors/glove.twitter.27B.100d.magnitude")
elmo_vecs = Magnitude("vectors/elmo_2x1024_128_2048cnn_1xhighway_weights.magnitude")
tokenizer = Tokenizer(num_words = 300)

tfidf = TfidfVectorizer(max_features = 300)
embed = hub.load('4')




def pipeline_lstm_feature(text):
    return pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=15).reshape(-1)


def pipeline_avg_glove(text):
    return np.average(glove.query(word_tokenize(text)), axis = 0)


def pipeline_idf_glove(text):
    idf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
    glove_vectors = glove.query(word_tokenize(text))
    weights = [idf_dict.get(word, 1) for word in word_tokenize(text)]
    return np.average(glove_vectors, axis = 0, weights = weights)


def pipeline_elmo(text):
    return elmo_vecs.query(text)


def pipeline_tfidf_vectorize(text):
    return tfidf.transform([text]).toarray().reshape(-1)


def read_data(dataset_path):
    print('Reading Data...')
    data = pd.read_csv(dataset_path)
    X, y = data.data.values, data.intent.values
    return X, y

def pipeline_sent_enc(text):
    return embed([text]).numpy()[0]


if __name__ == '__main__':
    dataset = read_data('dataset/mainModel.csv')
    dataset = preprocess(dataset, {
        'tfidf' : tfidf,
        'tokenizer' : tokenizer})
        
    features = {
       # 'lstm_features' : pipeline_lstm_feature,
        'sent_enc' : pipeline_sent_enc,
       # 'glove' : pipeline_avg_glove,
       # 'idf_glove' : pipeline_idf_glove,
       # 'tfidf' : pipeline_tfidf_vectorize,
       # 'elmo' : pipeline_elmo
    }

    features = featurize_and_split(dataset, features)
    """

    rf = {
        'model' : OneVsRestClassifier(RandomForestClassifier()),
        'params' : {'estimator__n_estimators' : [10, 50, 100]}
    }



    lstm = {
        'model' : LSTM_Model(tokenizer),
        'params' : 'validate'
    }
    """
    xgb = {
        'model' : MultiOutputClassifier(XGBClassifier(), n_jobs= -1),
        #'params' : {'estimator__max_depth' : [2,5,7], 'estimator__n_estimators': [100]}
        'params' : None,
        'features_to_run' : ['sent_enc', 'idf_glove']

    }

    svm = {
        'model' : OneVsRestClassifier(SVC(verbose = True, probability = True), n_jobs = -1),
        #'params' : {'estimator__C' : [0.1, 1, 10, 50, 100], 'estimator__kernel': ['rbf', 'linear']}
        'params' : None,
        'features_to_run' : ['sent_enc', 'idf_glove']

    }

    dnn = {
        'model' : DNN_Model(),
        'params' : 'validate',
        'features_to_run' : ['sent_enc']
    }
    cnn = {
        'model' : CNN_Model(tokenizer),
        'params' : 'validate',
        'features_to_run' : ['lstm_features']
    }


    svm_sgd = {
        'model' : OneVsRestClassifier(SGDClassifier(loss = 'log'), n_jobs = -1),
        'params' : {'estimator__alpha' : [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]},
        'features_to_run' : ['tfidf', 'idf_glove', 'sent_enc', 'glove']

    } 


    models = {
     #  'cnn' : cnn,
       'DNN' : dnn,
     #  'log_reg' : svm_sgd,
       #'svm' : svm
       #'rf' : rf
    }


    report(models, features)
  
