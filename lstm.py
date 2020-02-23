from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import SpatialDropout1D, Input, Dense, Embedding, LSTM, concatenate, Flatten, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam,RMSprop, Nadam
from tqdm import tqdm 
from pymagnitude import Magnitude
import numpy as np

class LSTM_Model:
    def __init__(self, tokenizer):    

        embedding_matrix =  self._build_matrix(tokenizer)

        MAX_SEQUENCE_LENGTH = 30
        num_words = embedding_matrix.shape[0]

        input_comments = Input(shape = (MAX_SEQUENCE_LENGTH,), name = 'user_utterance', dtype='int32')
        x = Embedding(num_words,
            100,
            embeddings_initializer = Constant(embedding_matrix),
            input_length = MAX_SEQUENCE_LENGTH,
            trainable = False)(input_comments)

        x = SpatialDropout1D(0.6)(x)
        x = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)
        #Output Layer
        x = Dense(3, activation="sigmoid", name = "Output")(x)
        self.model = Model(inputs=[input_comments],outputs = x)
        self.model.compile(loss = 'binary_crossentropy', metrics = ['acc'], optimizer = Adam(lr = 1e-4))

    def _build_matrix(self, tokenizer):
        vector = Magnitude('vectors/glove.twitter.27B.100d.magnitude')
        GLOVE_VECTOR_DIMENSION = 100
        MAX_NUM_WORDS = 300
        word_index = tokenizer.word_index
        num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
        embedding_matrix = np.zeros((num_words, GLOVE_VECTOR_DIMENSION))
        for word, i in tqdm(word_index.items()):
            if i > MAX_NUM_WORDS:
                continue
            embedding_vector = vector.query(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix



    def fit(self, x, y, x_test, y_test):
        self.model.fit(x, y,
            batch_size = 128,
            epochs = 40,
            verbose = 1,
            validation_data = (x_test, y_test))
        
    def predict(self, x):
        self.params['predict'](x)

    def predict_proba(self, x):
        return self.model.predict(x)