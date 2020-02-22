
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

