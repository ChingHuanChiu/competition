import sys
sys.path.append('./')

import keras
from data.data import *
import csv
import pickle

model = keras.models.load_model('./model/LSTM_CNN_model')
scaler = pickle.load(open('./model/daily_MinMax.pkl', 'rb'))


def model_strategy():
    """
    以每5根BAR資料傳進模型進行預測(因為本交易策略是利用5日資料預測隔天收盤是否上漲)，以貼近實際情況，並將預測結果(交易訊號）寫入CSV檔

    """
    with open('strategy/signal.csv', 'w', newline='', encoding='utf-8-sig') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['date', 'open', 'high', 'low', 'close', 'signal', 'etf', 'time'])
        for bar in get_bar_data():

            bar_for_predict = scaler.fit_transform(bar.iloc[:, 3:].values.reshape(5, 20))

            bar_for_predict = bar_for_predict.reshape(1, 5, 20)

            signal = model.predict_classes(bar_for_predict)

            writer.writerow([str(bar.iloc[-1, 1]), bar.iloc[-1, 3], bar.iloc[-1, 4], bar.iloc[-1, 5], bar.iloc[-1, 6],
                             signal[0], bar.iloc[-1, 0], bar.iloc[-1, 1]])




