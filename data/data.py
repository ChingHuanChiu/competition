
import pandas as pd
import glob
import talib

from collections import defaultdict


class Data:
    def __init__(self, dataset):
        self.dataset = dataset

    def _get_0050_0056_match_data(self):
        if self.dataset == 'Test':
            match_file = glob.glob('./Test_Data/ETF/*_Match.txt')  # list
        elif self.dataset == 'Train':
            match_file = glob.glob('./Train_Data/ETF/*_Match.txt')
        date_list = [date[-18:-10] for date in match_file]

        match_df = pd.DataFrame()
        for f, date in zip(match_file, date_list):
            data = pd.read_csv(f, header=None, converters={1: str})

            data['成交日期'] = pd.to_datetime([date] * len(data))
            match_df = pd.concat([data, match_df], axis=0)

        match_df.columns = ['成交時間', '商品代碼', '成交價格', '成交量', '累計成交量', '成交日期']
        match_df = match_df.sort_values(['成交日期', '成交時間'], ascending=True)
        df_0050 = match_df[match_df['商品代碼'] == '0050']
        df_0056 = match_df[match_df['商品代碼'] == '0056']
        return df_0050, df_0056

    def get_daily_etf_ohlc(self, prod):
        if prod == '0050':
            df = self._get_0050_0056_match_data()[0]
        elif prod == '0056':
            df = self._get_0050_0056_match_data()[1]

        ohlc_dict = defaultdict(list)
        for data in df.groupby('成交日期'):
            ohlc_dict['商品'].append(prod)
            ohlc_dict['成交日期'].append(data[0])
            ohlc_dict['開盤成交時間'].append(data[1]['成交時間'].iloc[0])
            ohlc_dict['開盤價'].append(data[1]['成交價格'].iloc[0])
            ohlc_dict['最高價'].append(max(data[1]['成交價格']))
            ohlc_dict['最低價'].append(min(data[1]['成交價格']))
            ohlc_dict['收盤價'].append(data[1]['成交價格'].iloc[-1])
            ohlc_dict['成交量'].append(data[1]['累計成交量'].iloc[-1])

            ohlc_dict["價格標準差"].append(data[1]['成交價格'].std())
            sum_price = 0
            for i in range(0, len(data[1]) - 1):
                sum_price += data[1]["成交價格"].iloc[i] * data[1]["成交量"].iloc[i]
                avg_price = sum_price / data[1]['累計成交量'].iloc[-2]
            ohlc_dict['加權股價'].append(avg_price)

        return pd.DataFrame(ohlc_dict)

    def get_half_hour_etf_ohlc(self, prod):
        if prod == '0050':
            df = self._get_0050_0056_match_data()[0]
        elif prod == '0056':
            df = self._get_0050_0056_match_data()[1]

        def get_subdata_ohlc(data):
            ohlc_dict = defaultdict()

            ohlc_dict['成交日期'] = data['成交日期'].iloc[0]
            ohlc_dict['成交時間'] = data['成交時間'].iloc[0]

            ohlc_dict['開盤價'] = data['成交價格'].iloc[0]
            ohlc_dict['最高價'] = max(data['成交價格'])
            ohlc_dict['最低價'] = min(data['成交價格'])
            ohlc_dict['收盤價'] = data['成交價格'].iloc[-1]
            ohlc_dict['成交量'] = data['累計成交量'].iloc[-1]
            ohlc_dict["價格標準差"] = data['成交價格'].std()
            sum_price = 0
            for i in range(0, len(data) - 1):
                sum_price += data["成交價格"].iloc[i] * data["成交量"].iloc[i]
                avg_price = sum_price / data['累計成交量'].iloc[-2]
            ohlc_dict['加權股價'] = avg_price

            return pd.DataFrame(ohlc_dict, index=[0])

        start = [90000000000, 93000000000, 100000000000, 103000000000, 110000000000, 113000000000, 120000000000,
                 123000000000, 130000000000]
        end = [93000000000, 100000000000, 103000000000, 110000000000, 113000000000, 120000000000, 123000000000,
               130000000000, 133000000000]
        half_0050_data = pd.DataFrame()
        for date in list(df['成交日期'].unique()):
            _0050_data = df[df['成交日期'] == date]

            for s, e in zip(start, end):
                try:
                    half_hour_data = _0050_data[(_0050_data['成交時間'] >= s) & (_0050_data['成交時間'] < e)].reset_index(
                        drop=True)
                    half_ohlc = get_subdata_ohlc(half_hour_data)
                    half_0050_data = pd.concat([half_0050_data, half_ohlc], 0)
                except:
                    # print(date)
                    continue
        return half_0050_data


class AIData(Data):

    def __init__(self, dataset):
        super(AIData, self).__init__(dataset)

    def get_feature_data(self, freq, prod):
        """
        製作特徵
        freq：日Ｋ或30分Ｋ
        """
        if freq == 'daily':
            data = self.get_daily_etf_ohlc(prod)

        elif freq == 'halfhour':
            data = self.get_half_hour_etf_ohlc(prod)

        data["MA3"] = talib.MA(data["收盤價"], timeperiod=3)
        data["MA5"] = talib.MA(data["收盤價"], timeperiod=5)
        data["MA10"] = talib.MA(data["收盤價"], timeperiod=10)
        data["o-c"] = data["開盤價"] - data["收盤價"]
        data["DEMA3"] = talib.DEMA(data["收盤價"], timeperiod=3)
        data["DEMA5"] = talib.DEMA(data["收盤價"], timeperiod=5)
        data["DEMA10"] = talib.DEMA(data["收盤價"], timeperiod=10)
        data["KAMA3"] = talib.KAMA(data["收盤價"], timeperiod=3)
        data["KAMA5"] = talib.KAMA(data["收盤價"], timeperiod=5)
        data["KAMA10"] = talib.KAMA(data["收盤價"], timeperiod=10)
        data["OBV"] = talib.OBV(data["收盤價"], data["成交量"])
        data["BETA"] = talib.BETA(data["最高價"], data["最低價"])
        data["VAR"] = talib.VAR(data["收盤價"], timeperiod=5, nbdev=1)
        data = data.dropna(axis=0, how='any')
        data = data.reset_index(drop=True)
        return data


def get_bar_data():
    """
        傳送５根K棒資料給strategy.py中的模型
    """
    _0050_data = pd.read_csv('data/daily_test_0050.csv')
    _0056_data = pd.read_csv('data/daily_test_0056.csv')
    for data in [_0050_data, _0056_data]:
        for i in range(9, len(data) + 1):
            bar = data.iloc[i - 5: i]
            yield bar


if __name__ == '__main__':
    import time
    # 製作資料
    s = time.time()

    # Train 的日資料
    daily_train = AIData('Train').get_feature_data('daily', '0050')
    half_hour_train = AIData('Train').get_feature_data('halfhour', '0050')
    # Test 的日資料
    daily_test = AIData('Test').get_feature_data('daily', '0050')
    daily_0056_test = AIData('Test').get_feature_data('daily', '0056')
    # Test 的30分K 資料
    half_hour_test = AIData('Test').get_feature_data('halfhour', '0050')


    # 輸出成CSV檔
    # daily_train.to_csv('daily_train_0050.csv', encoding='utf-8-sig', index=False)
    # daily_test.to_csv('daily_test_0050.csv', encoding='utf-8-sig', index=False)
    # daily_0056_test.to_csv('daily_test_0056.csv', encoding='utf-8-sig', index=False)

    # half_hour_train.to_csv('half_hour_train_0050.csv', encoding='utf-8-sig', index=False)
    # half_hour_test.to_csv('half_hour_test_0050.csv', encoding='utf-8-sig', index=False)
    e = time.time()
    print(e-s)
