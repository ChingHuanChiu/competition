
from collections import defaultdict
import pandas as pd
import numpy as np

from strategy.strategy import model_strategy
from KPI import KPI


class Action:
    """
    針對買賣條件紀錄交易內容
    """
    def __init__(self, trading_unit, stop_loss_rate):
        self.data_dict = defaultdict(list)
        self.position = 0
        self.stop_loss_rate = stop_loss_rate
        self.stop_loss_price = 0
        self.unit = trading_unit

    def long(self, date, time, price):
        """
        :param date: Buy date
        :param time: Buy time
        :param price: Buy price

        """
        self.data_dict['進場日期'].append(date)
        self.data_dict['進場時間'].append(time)
        self.data_dict['進場價格'].append(price)
        self.position += 1 * self.unit
        self.stop_loss_price = price * (1 - self.stop_loss_rate) if self.stop_loss_rate is not None else 0

    def short(self, date, time, price, p):
        """

        :param date: Sell date
        :param time: Sell time
        :param price: Sell price
        :param p: product
        """
        self.data_dict['商品代碼'].append(p)
        self.data_dict['出場日期'].append(date)
        self.data_dict['出場時間'].append(time)
        self.data_dict['出場價格'].append(price)
        self.data_dict['單位'].append(self.unit)

        self.position -= 1 * self.unit


def process_trading(data, signal, trading_unit, stop_loss_rate):
    assert len(data) == len(signal), 'Length of data must equal to the length of signal data'

    # 初始部位設為0
    position_li = [0]

    act = Action(trading_unit, stop_loss_rate)

    for i in range(len(data)-1):
        # 當今天部位為0且訊號為上漲，買進
        if act.position == 0 and signal[i] == 1:
            act.long(data.date.values[i+1], data.time.values[i+1], data.open.values[i+1])

        # 當今天部位不為0且訊號為下跌，賣出
        elif act.position > 0 and signal[i] == 0:
            act.short(data.date.values[i+1], data.time.values[i+1], data.open.values[i+1], data.etf.values[i+1])

        # 如果收盤價觸及停損價格，隔天開盤賣出
        elif data.close.values[i] <= act.stop_loss_price and act.position > 0:
            print(123)
            act.short(data.date.values[i+1], data.time.values[i+1], data.open.values[i+1], data.etf.values[i+1])


    # 當最後一筆資料部位大於0則強迫清倉
    if act.position > 0:
        act.short(data.date.values[-1], data.time.values[-1], data.open.values[-1], data.etf.values[-1])

        position_li.append(act.position)
    res = pd.DataFrame(act.data_dict)

    res['BS'] = np.where((res['進場日期'] < res['出場日期']) | (res['進場時間'] < res['出場時間']), 'B', 'S')
    return res[['商品代碼', 'BS', '進場日期', '進場時間', '進場價格', '出場日期', '出場時間', '出場價格', '單位']]


if __name__ == '__main__':
    import time
    # 進行test data預測，並產生訊號資料檔(CSV)
    sa = time.time()
    model_strategy()
    se = time.time()
    print('模型預測並產生訊號所花費時間：', se - sa, '秒')

    # 讀進訊號資料，開始紀錄交易
    sb = time.time()
    res = pd.DataFrame()
    df = pd.read_csv('./strategy/signal.csv')
    for data in df.groupby('etf'):
        data = data[1].reset_index(drop=True)
        # 交易單位1張(1000股)，且以買進價格的8％為停損點
        record = process_trading(data, data.signal, 1000, stop_loss_rate=0.08)
        res = pd.concat([res, record], 0).sort_values(['進場日期'], ascending=True)
    res.to_csv('Record.csv', index=False, encoding='utf-8-sig')
    eb = time.time()
    print('紀錄交易所花費時間：', eb - sb, '秒')

    # 開始回測，並記錄回測績效
    sc = time.time()
    KPI()
    ec = time.time()
    print('計算回測績效所花費時間：', ec - sc, '秒')
