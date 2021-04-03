import numpy as np
import pandas as pd
import pickle

from keras.models import Sequential
from keras.layers import LSTM, MaxPooling1D
from keras.utils.np_utils import *
from keras.optimizers import Adam
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.engine.topology import Layer
from tensorflow.keras.layers import Lambda, dot, concatenate
from keras.models import Model
from keras.layers import Conv1D, Activation, Dense
import tensorflow as tf
tf.compat.v1.set_random_seed(666)


class Attention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, hidden_states):
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28.
        """
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector


def LSTM_CNN(X_train, y_train, batch_size, epochs):
    input_shape = X_train[0].shape

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, padding="causal", activation='relu', input_shape=input_shape))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding="causal"))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding="causal"))
    model.add(MaxPooling1D())

    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(Attention())
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu', name='dense_1'))
    model.add(Dense(64, activation='relu', name='dense_2'))
    model.add(Dense(2, activation='softmax', name='pred'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    print('Start training......')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_split=0.2)
    print('---Finish training---')
    return model


def make_30K_train_data(lookback_win):
    """
    make the training data from 0050  that time frequency is 30 minutes
    """
    train = pd.read_csv('../data/half_hour_train_0050.csv')

    train = train.drop(['成交日期', '成交時間'], 1)
    train['return'] = train.收盤價.pct_change()

    # 當今日收盤價大於隔日收盤價視為上漲，以“1“標記，反之，以”0”標記
    train['return'] = np.where(train['return'] > 0.0, 1, 0)
    train = train.dropna(0)

    # 製作Ｘ的訓練資料與y的訓練資料
    X = train.drop('return', 1)
    y = train['return']

    #針對Ｘ資料進行ＭinMax
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train = []
    y_train = []


    for i in range(lookback_win, len(train)):
        X_train.append(X[i - lookback_win: i])
        y_train.append(y[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train = to_categorical(y_train, 2)
    return X_train, y_train


def make_daily_train_data(lookback_win):
    train_0050 = pd.read_csv('../data/daily_train_0050.csv')

    train_0050 = train_0050.drop(['成交日期', '開盤成交時間', '商品'], 1)
    train_0050['return'] = train_0050.收盤價.pct_change()
    train_0050['return'] = np.where(train_0050['return'] > 0.0, 1, 0)
    train_0050 = train_0050.dropna(0)

    ft_X = train_0050.drop('return', 1)
    ft_y = train_0050['return']

    scaler = MinMaxScaler()
    scaler.fit_transform(ft_X)
    ft_X = scaler.transform(ft_X)
    # 將ＭinMax 存成模型，以供之後進行真實資料（測試資料）的處理
    pickle.dump(scaler, open('daily_MinMax.pkl', 'wb'))

    ft_X_train = []
    ft_y_train = []

    for i in range(lookback_win, len(train_0050)):
        ft_X_train.append(ft_X[i - lookback_win: i])
        ft_y_train.append(ft_y[i])

    ft_X_train = np.array(ft_X_train)
    ft_y_train = np.array(ft_y_train)
    ft_y_train = to_categorical(ft_y_train, 2)
    return ft_X_train, ft_y_train


def fine_tune_model(pre_model, X_train, y_train, batch_size, epochs):
    print('----------------origin model------------------------')
    print(pre_model.summary())
    print('----------------------------------------------------')
    model = Model(inputs=pre_model.input, outputs=pre_model.layers[-2].output)
    print('----------------pre-train model---------------------')
    print(model.summary())
    print('----------------------------------------------------')

    # 搭建 fine tune 模型
    fc = Sequential()
    fc.add(model)
    fc.add(Dense(32, activation='relu'))
    fc.add(Dropout(0.5))
    fc.add(Dense(2, activation='softmax'))

    # 將CNN與LSTM訓練好的權重凍結，因此重訓練時並不會訓練被凍結的權重
    model.trainable = False
    print("=================Fine tune model===============")
    fc.summary()
    print('================================================')
    fc.compile(loss='binary_crossentropy',
               optimizer=Adam(learning_rate=0.00002),
               metrics=['accuracy'])
    fc.fit(X_train, y_train, batch_size, epochs, verbose=1,
           validation_split=0.2)

    return fc


if __name__ == '__main__':
    train_X_30K, train_y_30K = make_30K_train_data(lookback_win=5)

    m = LSTM_CNN(train_X_30K, train_y_30K, batch_size=128, epochs=50)

    fc_X_train, fc_y_train = make_daily_train_data(lookback_win=5)


    fc_model = fine_tune_model(m, fc_X_train, fc_y_train, batch_size=32, epochs = 50)

    fc_model.save('LSTM_CNN_model')







