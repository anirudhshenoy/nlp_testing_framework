
# Readme 


The framework requires 2 items to be defined: 

   * Feature Pipeline 
   * Model


## Feature Pipeline

For each feature that you want to test define a function as follows: 

``` 
    def pipeline_word_2_vec(text):
        # Here define how 'text' will get vectorized
        # eg : word_vector = vectors.query(text)
        return word_vector
```

* The function must take `text` as an argument 
* The function must return a single vectorized feature 
* Function can have any name 

### Feature Dictionary 

Once the feature functions have been defined create a dict as follows for each feature that you want to test: 

```
features = {
        'word2vec' : pipeline_word_2_vec,
        'sent_enc' : pipeline_sent_enc,
        'glove' : pipeline_avg_glove,
        'idf_glove' : pipeline_idf_glove,
        'tfidf' : pipeline_tfidf_vectorize,
        'elmo' : pipeline_elmo
    }
```

## Model Definition

For each model that you want to test create a dict as follows: 

```
    dnn = {
        'model' : DNN_Model(),
        'params' : 'validate',
        'features_to_run' : ['sent_enc']
    }
```

* `features_to_run` defines which features to try for this model. In this case, `sent_enc` feature will be tested for this model. 
* Set the `model` key to the instance of the model you want to test. 
* `params` takes 3 options: 
    * `validate` - Will run validation with the test set (only applicable for Keras Models)
    * `None` - No validation
    * A parameters dict to pass to `GridSearchCV` (only sklearn models)

### Sklearn Models 
For `sklearn` models you can directly pass the model instance to the dict. Eg: 

```
    dnn = {
        'model' : OneVsRestClassifier(SVC()),
        'params' : {'estimator__C' : [0.1, 1, 10, 50, 100]},
        'features_to_run' : ['sent_enc']
    }
```

* `OneVsRestClassifier()` converts the `SVC` model to multi-labeling.
* When this model is run `GridSearchCV` will try different values of `C` as defined in the `params` parameter.

### Keras Models

For Keras models a simple wrapper needs to be implemented. Check the `dnn.py` file for an example. 

* In the `__init__()` class initializer you can define the model architecture and compile it.
* The `fit()` function will be called during testing with `(X_train, y_train, X_test, y_test)`. In `fit()` define how your model will be trained with this data. 
* `predict_proba()` will be called with `X` (single datapoint). Here your model needs to return the probability for each class. For keras models its a simple `self.model.predict(X)`

Once the wrapper is created you can import into `model.py` and define the model dict as follows: 

```
    dnn = {
        'model' : DNN_Model(),
        'params' : 'validate',
        'features_to_run' : ['sent_enc']
    }
```


## Models Dict
For each model that you have defined and want to test add it to the `models` dict as follows: 
```
models = {
       'cnn' : cnn,
       'DNN' : dnn,
       'log_reg' : svm_sgd,
       'svm' : svm,
       'rf' : rf
    }
```

## Generating the Report

Finally call `report(models, features)` to start the testing process. 

# Dependencies 
* tqdm
* pandas
* numpy
* prettytable

Install with  `pip3 install --user tqdm pandas numpy prettytable` 

#### Tensorflow Bug with python 3.7.3

TF 2.0 seems to have a big with Python 3.7.3 on Macs. Upgrade to Python 3.7.6 and reinstall tf 2.0 if you have any issues. 