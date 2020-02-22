from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def featurize_and_split(dataset, features, filepath = 'dataset/training-set.csv', split_type = 'file'):
    print('Featurizing...')

    X, y = dataset
    (X_train, y_train), (X_test, y_test) = split_dataset(X, y, 
                    split_type = 'file', filepath = filepath)

    le = LabelBinarizer()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    for feature_name, feature in features.items():
        X_train_feat, X_test_feat = feature((X_train, X_test))
        features[feature_name] = {
            'featurizer' : feature,
            'feature_data' : ((X_train_feat, y_train), (X_test_feat, y_test))
        }
    return features 

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
    (X_test, y_test) = (test_data.data.values, test_data.intent.values)
    return (X_test, y_test)

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