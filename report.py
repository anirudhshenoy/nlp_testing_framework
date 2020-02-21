from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score, precision_recall_curve, precision_score, recall_score
import numpy as np 
from prettytable import PrettyTable

class Model:
    def __init__(self, params):
        self.params = params 

    def fit(self, x, y):
        self.params['fit'](x, y)

    def predict(self, x):
        self.params['predict'](x)

    def predict_proba(self, x):
        self.params['predict_proba'](x)

def train(x, y):
    print('training.....')

def predict(x):
    print('predicting.......')


def calc_f1(p_and_r):
    p, r = p_and_r
    return (2*p*r)/(p+r)

def print_table(results):
    x = PrettyTable()
    x.field_names = ["Model", "F1", "AUC", "Accuracy"]

    for model_name, result in results.items(): 
        result = [round(r, 2) for r in result]
        x.add_row([model_name] + result)
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
    pr = precision_score(y_test, y_test_pred, average = 'samples')
    f1 = f1_score(y_test, y_test_pred, average = 'samples')
    roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class= 'ovo')
    acc = accuracy_score(y_test, y_test_pred)
        
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(f1, best_precision, best_recall, roc_auc, acc))
    
    if return_metrics:
        return [f1, roc_auc, acc]
    

def report(models, train_data, test_data):

    (X_train, y_train), (X_test, y_test) = train_data, test_data

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        y_pred = np.array([y[:,1] for y in y_pred]).transpose()
        results[model_name] = print_model_metrics(y_test, y_pred)

    print_table(results)
if __name__ == '__main__':

    dc = Model({
        'fit' : train,
        'predict' : predict
    })

    dc.fit(1,2)
    dc.predict(2)
