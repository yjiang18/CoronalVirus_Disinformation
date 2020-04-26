from tensorflow.keras.layers import Input, Dense, BatchNormalization,Conv1D,Activation,MaxPooling1D,Concatenate, Flatten
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Model
from keras.engine import Layer
from keras import initializers
from keras import backend as K

from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, GRU, Bidirectional, TimeDistributed
from keras.models import Model

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim, )), name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])



class Ye_CNN():
    def __init__(self):
        self.model = None

    def cnn_model(self, tok_len, embed_size):
        filter_sizes = [1, 2, 3, 4, 5, 6, 7]
        num_filters = 128
        sent_input = Input(shape=(tok_len, embed_size),dtype='float')
        conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]))(sent_input)
        act_0 = Activation('relu')(conv_0)
        bn_0 = BatchNormalization()(act_0)

        conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]))(sent_input)
        act_1 = Activation('relu')(conv_1)
        bn_1 = BatchNormalization()(act_1)

        conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]))(sent_input)
        act_2 = Activation('relu')(conv_2)
        bn_2 = BatchNormalization()(act_2)

        conv_3 = Conv1D(num_filters, kernel_size=(filter_sizes[3]))(sent_input)
        act_3 = Activation('relu')(conv_3)
        bn_3 = BatchNormalization()(act_3)

        conv_4 = Conv1D(num_filters, kernel_size=(filter_sizes[4]))(sent_input)
        act_4 = Activation('relu')(conv_4)
        bn_4 = BatchNormalization()(act_4)

        conv_5 = Conv1D(num_filters, kernel_size=(filter_sizes[5]))(sent_input)
        act_5 = Activation('relu')(conv_5)
        bn_5 = BatchNormalization()(act_5)

        conv_6 = Conv1D(num_filters, kernel_size=(filter_sizes[6]))(sent_input)
        act_6 = Activation('relu')(conv_6)
        bn_6 = BatchNormalization()(act_6)

        maxpool_0 = MaxPooling1D(pool_size=(tok_len - filter_sizes[0]))(bn_0)
        maxpool_1 = MaxPooling1D(pool_size=(tok_len - filter_sizes[1]))(bn_1)
        maxpool_2 = MaxPooling1D(pool_size=(tok_len - filter_sizes[2]))(bn_2)
        maxpool_3 = MaxPooling1D(pool_size=(tok_len - filter_sizes[3]))(bn_3)
        maxpool_4 = MaxPooling1D(pool_size=(tok_len - filter_sizes[4]))(bn_4)
        maxpool_5 = MaxPooling1D(pool_size=(tok_len - filter_sizes[5]))(bn_5)
        maxpool_6 = MaxPooling1D(pool_size=(tok_len - filter_sizes[6]))(bn_6)

        sent_concat = Concatenate()([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4,
                                     maxpool_5, maxpool_6])
        sent_flat = Flatten()(sent_concat)
        sent_den = Dense(units=100, activation='relu')(sent_flat)
        pred = Dense(12, activation='softmax')(sent_den)

        self.model = Model(sent_input, pred)
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', metrics=['acc', precision_m, recall_m, f1_m],
                           optimizer='adam')

    def build_model(self, max_tok, embed_size):
        self.cnn_model(max_tok, embed_size)

    def train(self, x_train, y_train, x_val, y_val, batch_size, checkpoint=None):
        self.model.fit(x_train, y_train, epochs=30, batch_size=batch_size, validation_data=[x_val, y_val],
                       verbose=1, callbacks=[checkpoint])

        loss, accuracy, precision, recall, f1_score = self.model.evaluate(x_val, y_val, verbose=1)

        return loss, accuracy, precision, recall, f1_score

class Ye_RNN_ATT():
    def __init__(self):
        self.model = None

    def rnn_model(self, tok_len, embed_size):
        sent_input = Input(shape=(tok_len, embed_size), dtype='float')
        sent_lstm = Bidirectional(GRU(100, return_sequences=True))(sent_input)
        att = AttLayer(100)(sent_lstm)
        pred = Dense(12, activation='softmax')(att)
        self.model = Model(sent_input, pred)
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', metrics=['acc', precision_m, recall_m, f1_m],
                           optimizer='adam')

    def build_model(self, max_tok, embed_size):
        self.rnn_model(max_tok, embed_size)

    def train(self, x_train, y_train, x_val, y_val, batch_size, checkpoint=None):
        self.model.fit(x_train, y_train, epochs=30, batch_size=batch_size, validation_data=[x_val, y_val],
                       verbose=1, callbacks=[checkpoint])

        loss, accuracy, precision, recall, f1_score = self.model.evaluate(x_val, y_val, verbose=1)

        return loss, accuracy, precision, recall, f1_score

class Ye_HAN():
    def __init__(self):
        self.model = None

    def han_model(self, seq_len, tok_len, embed_size):
        sentence_input = Input(shape=(seq_len, embed_size), dtype='float')
        l_lstm = Bidirectional(GRU(100, return_sequences=True))(sentence_input)
        l_att = AttLayer(100)(l_lstm)
        sentEncoder = Model(sentence_input, l_att)

        review_input = Input(shape=(seq_len, tok_len, embed_size), dtype='float')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
        l_att_sent = AttLayer(100)(l_lstm_sent)
        preds = Dense(11, activation='softmax')(l_att_sent)
        self.model = Model(review_input, preds)

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc', precision_m, recall_m, f1_m])

    def build_model(self,seq_len, max_tok, embed_size):
        self.han_model(seq_len, max_tok, embed_size)

    def train(self, x_train, y_train, x_val, y_val, batch_size, checkpoint=None):
        self.model.fit(x_train, y_train, epochs=30, batch_size=batch_size, validation_data=[x_val, y_val],
                       verbose=1, callbacks=[checkpoint])

        loss, accuracy, precision, recall, f1_score = self.model.evaluate(x_val, y_val, verbose=1)

        return loss, accuracy, precision, recall, f1_score
