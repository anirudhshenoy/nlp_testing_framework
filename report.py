from sklearn.metrics import log_loss, f1_score, roc_auc_score, accuracy_score, precision_recall_curve, precision_score, recall_score, confusion_matrix, hamming_loss
import numpy as np 
from prettytable import PrettyTable
from sklearn.model_selection import GridSearchCV

def print_table(results):
    x = PrettyTable()
    x.field_names = ["Name", "Model", "Feature", "F1", "AUC", "Hamming", "Log Loss"]

    for model_name, result in results.items(): 
        result['metrics'] = [round(r, 2) for r in result['metrics']]
        x.add_row([model_name]+ [result['model_name']] + [result['feature']] + result['metrics'])
    print(x)

    Html_file= open("report.html","w")
    Html_file.write(x.get_html_string())
    Html_file.close()


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
    

    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(f1, best_precision, best_recall, roc_auc, acc))
    
    if return_metrics:
        return [f1, roc_auc, hamming, loss]

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
            (X_train, y_train), (X_test, y_test) = feature['feature_data']
            model = run_grid_search(model['model'], model['params'], X_train, y_train)
            y_pred = model.predict_proba(X_test)
            y_pred = np.array([y[:,1] for y in y_pred]).transpose()
            temp_results = print_model_metrics(y_test, y_pred)
            results[model_name + '_' + feature_name] = {
                'model_name' : model_name,
                'metrics' : temp_results,
                'feature' : feature_name,
                'model' : model
            }

    print_table(results)


if __name__ == '__main__':

    dc = Model({
        'fit' : train,
        'predict' : predict
    })

    dc.fit(1,2)
    dc.predict(2)
