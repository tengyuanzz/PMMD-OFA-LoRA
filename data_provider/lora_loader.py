import os
import pandas as pd
from torch.utils.data import Dataset

from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler

class Dataset_Lora(Dataset):
    def __init__(self, root_path, size=None,
                lora_path='aapl_ohclv_08.csv',
                technical_indicators = False,
                target='OT', scale=True, percent=10):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init


        self.target = target
        self.scale = scale
        self.percent = percent
        self.technical_indicators = technical_indicators
        self.rolling_window = 14

        self.root_path = root_path
        self.lora_path = lora_path
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        print('reading for lora 08 data')
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,  self.lora_path))
        if self.lora_path == 'aapl_sent_08.csv':
            df_raw.columns = ["date", "Open", "High", "Low", "Close", "Volume", "Neutral","Positive","Negative"]
        else:
            df_raw.columns = ["date", "Open", "High", "Low", "Close", "Volume"]
        df_raw['date'] = pd.to_datetime(df_raw['date'], utc=True).dt.tz_convert(None)
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]] # target variable is the last index

        if self.technical_indicators:
            print('Using Technical indicators')
            df_raw[f'MA_{self.rolling_window}'] = df_raw[self.target].rolling(window=self.rolling_window).mean()
            from ta.momentum import RSIIndicator
            df_raw['RSI'] = RSIIndicator(df_raw[self.target]).rsi()
            from ta.trend import MACD
            macd = MACD(df_raw[self.target])
            df_raw['MACD'] = macd.macd()
            from ta.volatility import BollingerBands
            bb = BollingerBands(df_raw[self.target])
            df_raw['BB_high'] = bb.bollinger_hband()
            df_raw['BB_low'] = bb.bollinger_lband()

        df_raw = df_raw.dropna().reset_index(drop=True)

        border1 = 0
        border2 = len(df_raw)
        # border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1:border2]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values



        self.data_x = data[border1:border2]
        # print('self.data_x',self.data_x.shape)
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # debug print
        print(f"[GETITEM] idx={index} → feat_id={feat_id}, "
              f"s_begin={s_begin}, s_end={s_end}, "
              f"r_begin={r_begin}, r_end={r_end}", flush=True)

        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]


        return seq_x, seq_y

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)