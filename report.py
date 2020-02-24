from sklearn.metrics import log_loss, f1_score, roc_auc_score, accuracy_score, precision_recall_curve, precision_score, recall_score, confusion_matrix, hamming_loss
import numpy as np 
import pandas as pd
from prettytable import PrettyTable
from sklearn.model_selection import GridSearchCV
from utility import read_test_file
from sklearn.preprocessing import LabelBinarizer
import spacy
nlp = spacy.load('en_core_web_sm')

def negation_model(text):
    doc = nlp(text)
    return [tok for tok in doc if tok.dep_ == 'neg']

def print_table(results):
    x = PrettyTable()
    x.field_names = ["Name", "Model", "Feature", "F1", "AUC", "Hamming", "Log Loss"]
    html = '<h2> Least Confident Utterances </h2> \n' 

    for model_name, result in results.items(): 
        result['metrics'] = [round(r, 2) for r in result['metrics']]
        x.add_row([model_name]+ [result['model_name']] + [result['feature']] + result['metrics'])
        print(model_name)
        html += '<h3>' + model_name + '</h3>\n'
        html += result['least_conf'].to_html()
    print(x)

    Html_file= open("report.html","w")
    Html_file.write('<h1> Model Testing Report </h1> \n')
    Html_file.write('<h3> Overview </h3> \n')
    ptable = x.get_html_string()
    ptable = ptable[:6] + ' border = "1"' +ptable[6:]
    Html_file.write(ptable)
    Html_file.write(html)
    Html_file.close()

def safe_ln(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))

def least_confident(targets, preds):
    X, y = read_test_file('dataset/training-set.csv')
    ce = -np.sum(targets * safe_ln(preds), axis = 1)
    max_index = np.flip(np.argsort(ce))
    rows = []
    for idx in max_index:
        row = []
        row.append(X[idx])
        row.append(y[idx])
        row = row + preds[idx].tolist()
        row.append(ce[idx])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.columns = ['Utterance', 'True Intent'] + np.unique(y).tolist() + ['Loss']
    return df


def print_model_metrics(y_test, y_pred_prob, verbose = False, return_metrics = True):
    best_threshold = 0.5
    #precision, recall, threshold = precision_recall_curve(y_test, y_pred_prob, pos_label = 1)
    
    #Find the threshold value that gives the best F1 Score
    #best_f1_index =np.argmax([calc_f1(p_r) for p_r in zip(precision, recall)])
    #best_threshold, best_precision, best_recall = threshold[best_f1_index], precision[best_f1_index], recall[best_f1_index]

    # Calulcate predictions based on the threshold value
    y_test_pred = np.where(y_pred_prob > best_threshold, 1, 0)
    
    # Calculate all metrics
    #pr = precision_score(y_test, y_test_pred, average = 'samples')
    f1 = f1_score(y_test, y_test_pred, average = 'samples')
    roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class= 'ovr')
    #acc = accuracy_score(y_test, y_test_pred)
    hamming = hamming_loss(y_test, y_test_pred)
    loss = log_loss(y_test, y_pred_prob)
    least_conf = least_confident(y_test, y_pred_prob)

    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(f1, best_precision, best_recall, roc_auc, acc))
    
    if return_metrics:
        return ([f1, roc_auc, hamming, loss], least_conf)

def run_grid_search(model, params, features, y):
    grid = GridSearchCV(model, params, cv = 7, n_jobs = -1, scoring = 'f1_samples', verbose = 1, refit = True)    # Change to cv=7
    grid.fit(features,y)
    print(grid.best_params_)
    return grid.best_estimator_       
    

def report(models, features):
    results = {}
    for feature_name, feature in features.items():
        for model_name, model in models.items():
            #model.fit(X_train, y_train)
            if feature_name not in model['features_to_run']:
                continue
            (X_train, y_train), (X_test, y_test) = feature['feature_data']
            if model['params'] == 'validate':
                model = model['model']
                model.fit(X_train, y_train, X_test, y_test)
            elif model['params']:
                model = run_grid_search(model['model'], model['params'], X_train, y_train)
            else:
                model = model['model']
                model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)
            #y_pred = np.array([y[:,1] for y in y_pred]).transpose()
            temp_results, least_conf = print_model_metrics(y_test, y_pred)
            least_conf.to_csv('least_conf/' + model_name + '.csv')
            results[model_name + '_' + feature_name] = {
                'model_name' : model_name,
                'metrics' : temp_results,
                'feature' : feature_name,
                'featurizer' : feature['featurizer'],
                'class_labels' : feature['class_labels'],
                'model' : model,
                'least_conf' : least_conf,
            }

    print_table(results)
    run_test(results, features)

def run_test(models, features):
    while True:
        user_input = input()
        rows = []
        for model_name, model in models.items():
            input_feature = model['featurizer'](user_input).reshape(1,-1)
            probs = model['model'].predict_proba(input_feature)
            negative_words = negation_model(user_input)
            neg_flag = 'True' if negative_words else 'False'
            rows.append([model['model_name']] + [model['feature']] + [str(round(p,3)) for p in probs[0]] + [neg_flag])
        print_test_table(rows, model['class_labels'])

def print_test_table(rows, classes):
    x = PrettyTable()
    x.field_names = ['Model', 'Feature'] + list(classes) + ['Negation']
    for row in rows:
        x.add_row(row)
    print(x)
    print('\n')
   


if __name__ == '__main__':

    dc = Model({
        'fit' : train,
        'predict' : predict
    })

    dc.fit(1,2)
    dc.predict(2)
