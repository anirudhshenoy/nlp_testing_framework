from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import SpatialDropout1D, Input, Dense, Embedding, LSTM, concatenate, Flatten, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam,RMSprop, Nadam
from tqdm import tqdm 
from pymagnitude import Magnitude
import numpy as np

class DNN_Model:
    def __init__(self):    

        input_comments = Input(shape = (512,), name = 'user_utterance')
        x = Dense(256, activation = 'relu')(input_comments)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation = 'relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(3, activation="sigmoid", name = "Output")(input_comments)
        self.model = Model(inputs=[input_comments],outputs = x)

        self.model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

        print(self.model.summary())

    def fit(self, x, y, x_test, y_test):
        self.model.fit(x, y,
            batch_size = 64,
            epochs = 150,
            verbose = 1,
            validation_data = (x_test, y_test))
        
    def predict(self, x):
        self.params['predict'](x)

    def predict_proba(self, x):
        return self.model.predict(x)