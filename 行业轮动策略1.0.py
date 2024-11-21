import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from numpy import nan as NA
import matplotlib.pyplot as plt
from pandas.tseries.offsets import Day,MonthEnd,BDay
import os
from scipy import stats
from scipy.stats import percentileofscore
from scipy.stats import rankdata, mstats, zscore
from tqdm import tqdm
import empyrical as em
import itertools
from sklearn.preprocessing import QuantileTransformer

from iFinDPy import *


class Industry_Strategy_10Day:
    #全局属性
    def __init__(self):
        self.start_date = '2024-11-01'

        self.holding_period = 10

        self.factor_export_path=r"D:\量化交易构建\ETF轮动策略\行业轮动因子库\有效因子"
    
    #计算工具函数
    def add_holding_period_factors(self, df, holding_period, start_date=None):
        df['time'] = pd.to_datetime(df['time'])

        # 创建一个新列 `real_standard_factor`，初始复制 `standard_factor` 列
        df['real_standard_factor'] = df['standard_factor']

        # 如果提供了start_date，将其转换为datetime，以便比较
        if start_date:
            start_date = pd.to_datetime(start_date)

        # 对于每个股票代码，我们将每holding_period个交易日更新一次 `real_standard_factor`
        for thscode in tqdm(df['thscode'].unique()):
            # 选取当前股票的数据
            stock_data = df[df['thscode'] == thscode]

            # 如果设置了开始日期，仅选择开始日期之后的数据进行处理
            if start_date:
                stock_data = stock_data[stock_data['time'] >= start_date]

            # 计算每holding_period个交易日的第一个
            indices = stock_data.index[::holding_period]  # 每holding_period个交易日选择一个索引
            values = stock_data.loc[indices, 'standard_factor']  # 对应索引的factor值

            # 设置real_standard_factor，首先全部设置为NaN，然后填充选定的值
            df.loc[stock_data.index, 'real_standard_factor'] = None
            df.loc[indices, 'real_standard_factor'] = values
            # 填充NaN值，使用前向填充
            df.loc[stock_data.index, 'real_standard_factor'] = df.loc[stock_data.index, 'real_standard_factor'].ffill()

        return df

    def winsorize_and_standardize(self,df):
        # 使用标准差方法处理极端值
        def limit_to_std(x):
            # 计算平均值和标准差
            mean = x.mean()
            std = x.std()
            # 将超过3倍标准差的值设为3倍标准差
            x = x.apply(lambda v: min(max(v, mean - 3 * std), mean + 3 * std))
            return x

        # 应用limit_to_std函数处理每个时间组的因子值
        df['standard_factor'] = df['factor'].groupby(df['time']).transform(limit_to_std)

        # 使用zscore方法进行标准化
        df['standard_factor'] = df.groupby('time')['standard_factor'].transform(zscore)

        result = df[['time', 'thscode', 'factor', 'standard_factor']]

        return result

    def QuantileTransformer_and_standardize(self,df):
        # 检查并替换 inf 和 -inf 为 NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 删除包含 NaN 值的行
        df = df.dropna(subset=['factor'])

        # 创建QuantileTransformer对象，将输出转换为标准正态分布
        qt = QuantileTransformer(output_distribution='normal', random_state=0)
        
        # 对每个时间段的因子值进行分位数变换
        df['standard_factor'] = df.groupby('time')['factor'].transform(lambda x: qt.fit_transform(x.values.reshape(-1, 1)).flatten())
        df['standard_factor'] =zscore(df['standard_factor'])
        return df[['time', 'thscode', 'factor', 'standard_factor']]


    #因子计算程序

    #动量类
    def VWAP_Volume_Momentum_Factor(self, window_1=6):
        # 这里我们直接使用类的属性，而不是在参数列表中使用self
        holding_period = self.holding_period
        
        start_date = self.start_date

        # 计算因子的函数
        def calculate_factor(df, window):
            # 确保数据按照时间排序
            df = df.sort_values('time')

            # 计算每天的平均价格 (high + low + close) / 3
            df['avg_price'] = (df['high'] + df['low'] + df['close']) / 3

            # 计算VWAP
            df['vwap'] = (df['avg_price'] * df['volume']).cumsum() / df['volume'].cumsum()

            # 计算VWAP与收盘价的差值
            df['vwap_close_diff'] = df['vwap'] - df['close']

            # 计算3日内VWAP与收盘价差值的最大值和最小值
            df['vwap_close_diff_max'] = df['vwap_close_diff'].rolling(window=window).max()
            df['vwap_close_diff_min'] = df['vwap_close_diff'].rolling(window=window).min()

            # 对最大值和最小值进行排名
            df['rank_vwap_close_diff_max'] = df['vwap_close_diff_max'].rank(method='first')
            df['rank_vwap_close_diff_min'] = df['vwap_close_diff_min'].rank(method='first')

            # 计算3日内成交量变化
            df['delta_volume'] = df['volume'].diff(window)

            # 对成交量变化进行排名
            df['rank_delta_volume'] = df['delta_volume'].rank(method='first')

            # 计算因子值
            df['factor'] = (df['rank_vwap_close_diff_max'] + df['rank_vwap_close_diff_min']) * df['rank_delta_volume']

            return df

        # 计算所有股票的因子
        def calculate_all_factors(price_path, window):
            # 获取所有股票代码
            all_data_file = os.listdir(price_path)
            # 创建一个空的DataFrame来保存所有股票的因子
            all_factors = pd.DataFrame()

            # 遍历每一个股票代码，计算其因子，并添加到all_factors中
            for code in all_data_file:
                try:
                    df = pd.read_csv(price_path + '\\' + code)
                    # 对每一个数据都计算第一步定义的算法
                    factor = calculate_factor(df, window)
                    all_factors = all_factors.append(factor)
                except Exception as e:
                    print(code + '出现错误跳过:', e)

            return all_factors

        price_path = r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
        # 计算所有股票的因子，并且合并在一个表中
        all_factors = calculate_all_factors(price_path, window_1)

        all_factors = all_factors.dropna()

        # 对因子进行去除异常值和标准化操作
        all_factors = self.winsorize_and_standardize(all_factors)
        
        # 将结果保存到CSV文件中
        filename = 'VWAP' + str(window_1) + 'D'
        
        all_factors.reset_index(drop=True, inplace=True)

        # 使用self的holding_period和start_date参数
        result = self.add_holding_period_factors(all_factors, holding_period, start_date)

        result.to_csv(self.factor_export_path + "\\" + filename + '.csv', index=False)

        return result
    
    def Rank_Vwap_Factor(self):
        # 这里我们直接使用类的属性，而不是在参数列表中使用self
        holding_period = self.holding_period
        
        start_date = self.start_date

        # 计算因子的函数
        def calculate_factor(df):
            # 确保数据按照时间排序
            df = df.sort_values('time')

            # 计算VWAP (成交量加权平均价格)
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

            # 计算 (vwap - close) 的排名
            df['rank_vwap_close_diff'] = df['vwap'] - df['close']
            df['rank_vwap_close_diff'] = df['rank_vwap_close_diff'].rank()

            # 计算 (vwap + close) 的排名
            df['rank_vwap_close_sum'] = df['vwap'] + df['close']
            df['rank_vwap_close_sum'] = df['rank_vwap_close_sum'].rank()

            # 计算因子值
            df['factor'] = df['rank_vwap_close_diff'] / df['rank_vwap_close_sum']

            return df

        # 计算所有股票的因子
        def calculate_all_factors(price_path):
            # 获取所有股票代码
            all_data_file = os.listdir(price_path)
            # 创建一个空的DataFrame来保存所有股票的因子
            all_factors = pd.DataFrame()

            # 遍历每一个股票代码，计算其因子，并添加到all_factors中
            for code in all_data_file:
                try:
                    df=pd.read_csv(price_path+'\\'+code)
                    #对每一个数据都计算第一步定义的算法
                    factor = calculate_factor(df)
                    all_factors = all_factors.append(factor)
                except:
                    print(code+'出现错误跳过')

            return all_factors

        price_path = r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
        # 计算所有股票的因子，并且合并在一个表中
        all_factors = calculate_all_factors(price_path)

        all_factors = all_factors.dropna()

        # 对因子进行去除异常值和标准化操作
        all_factors = self.QuantileTransformer_and_standardize(all_factors)
        
        # 将结果保存到CSV文件中
        filename = 'Rank_Vwap'
        
        all_factors.reset_index(drop=True, inplace=True)

        # 使用self的holding_period和start_date参数
        result = self.add_holding_period_factors(all_factors, holding_period, start_date)

        result.to_csv(self.factor_export_path + "\\" + filename + '.csv', index=False)

        return result

    #反转类
    def LVVC(self, window_1=20, window_2=20):
        # 这里我们直接使用类的属性，而不是在参数列表中使用self
        holding_period = self.holding_period
        
        start_date = self.start_date

        # 计算因子的函数
        def calculate_factor(df, window_1=20, window_2=20):

            # 确保数据按照时间排序
            df = df.sort_values('time')

            # 计算过去10天最高价的标准差
            df['stddev_high'] = df['high'].rolling(window=window_1).std()

            # 对标准差进行排名
            df['rank_stddev_high'] = df['stddev_high'].rank(pct=True)

            # 计算最高价和成交量的10天相关性
            df['corr_high_volume'] = df['high'].rolling(window=window_2).corr(df['volume'])

            # 计算最终因子值
            df['factor'] = (-1 * df['rank_stddev_high']) * df['corr_high_volume']

            return df

        # 计算所有股票的因子
        def calculate_all_factors(price_path):
            # 获取所有股票代码
            all_data_file = os.listdir(price_path)
            # 创建一个空的DataFrame来保存所有股票的因子
            all_factors = pd.DataFrame()

            # 遍历每一个股票代码，计算其因子，并添加到all_factors中
            for code in all_data_file:
                try:
                    df = pd.read_csv(price_path + '\\' + code)
                    # 对每一个数据都计算第一步定义的算法
                    factor = calculate_factor(df)
                    all_factors = all_factors.append(factor)
                except Exception as e:
                    print(code + '出现错误跳过:', e)

            return all_factors

        price_path = r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
        # 计算所有股票的因子，并且合并在一个表中
        all_factors = calculate_all_factors(price_path)

        all_factors = all_factors.dropna()

        # 对因子进行去除异常值和标准化操作
        all_factors = self.winsorize_and_standardize(all_factors)
        
        # 将结果保存到CSV文件中
        filename = 'LVVC' + str(window_1) + 'D'
        
        all_factors.reset_index(drop=True, inplace=True)

        # 使用self的holding_period和start_date参数
        result = self.add_holding_period_factors(all_factors, holding_period, start_date)

        result.to_csv(self.factor_export_path + "\\" + filename + '.csv', index=False)

        return result
       #反转类
    
    def VRRF(self, window=5):
        # 这里我们直接使用类的属性，而不是在参数列表中使用self
        holding_period = self.holding_period
        
        start_date = self.start_date

        def calculate_factor(df, window=5):

            # 确保数据按照时间排序
            df = df.sort_values('time')

            # 对成交量进行排名
            df['rank_volume'] = df['volume'].rank(pct=True)

            # 计算过去5天内最高价和排名后的成交量之间的相关性
            df['corr_high_rank_volume'] = df['high'].rolling(window=window).corr(df['rank_volume'])

            # 计算最终因子值，乘以 -1
            df['factor'] = -1 * df['corr_high_rank_volume']

            return df
        # 计算所有股票的因子
        def calculate_all_factors(price_path):
            # 获取所有股票代码
            all_data_file = os.listdir(price_path)
            # 创建一个空的DataFrame来保存所有股票的因子
            all_factors = pd.DataFrame()

            # 遍历每一个股票代码，计算其因子，并添加到all_factors中
            for code in all_data_file:
                try:
                    df = pd.read_csv(price_path + '\\' + code)
                    # 对每一个数据都计算第一步定义的算法
                    factor = calculate_factor(df)
                    all_factors = all_factors.append(factor)
                except Exception as e:
                    print(code + '出现错误跳过:', e)

            return all_factors

        price_path = r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
        # 计算所有股票的因子，并且合并在一个表中
        all_factors = calculate_all_factors(price_path)

        all_factors = all_factors.dropna()

        # 对因子进行去除异常值和标准化操作
        all_factors = self.QuantileTransformer_and_standardize(all_factors)
        
        # 将结果保存到CSV文件中
        filename = 'VRRF' + str(5) + 'D'
        
        all_factors.reset_index(drop=True, inplace=True)

        # 使用self的holding_period和start_date参数
        result = self.add_holding_period_factors(all_factors, holding_period, start_date)

        result.to_csv(self.factor_export_path + "\\" + filename + '.csv', index=False)

        return result
   
    def PVCMRF(self):
        # 这里我们直接使用类的属性，而不是在参数列表中使用self
        holding_period = self.holding_period
        
        start_date = self.start_date

        def calculate_factor(df):
            # 确保数据按照时间排序
            df = df.sort_values('time')

            # 计算成交量加权平均价（VWAP）
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

            # 计算开盘价和最低价的加权平均值 (open * 0.178404) + (low * (1 - 0.178404))
            df['weighted_open_low'] = (df['open'] * 0.178404) + (df['low'] * (1 - 0.178404))

            # 计算过去 12 天的加权_open_low 的和
            df['sum_weighted_open_low'] = df['weighted_open_low'].rolling(window=12).sum()

            # 计算过去 120 天的平均成交量 (adv120)
            df['adv120'] = df['volume'].rolling(window=120).mean()

            # 计算过去 12 天 adv120 的总和
            df['sum_adv120'] = df['adv120'].rolling(window=12).sum()

            # 计算 sum_weighted_open_low 和 sum_adv120 的 16 天相关性
            df['correlation_price_volume'] = df['sum_weighted_open_low'].rolling(window=16).corr(df['sum_adv120'])

            # 对相关性结果进行排名
            df['rank_correlation'] = df['correlation_price_volume'].rank()

            # 计算 (high + low) / 2 和 VWAP 的加权平均值
            df['weighted_mid_vwap'] = (((df['high'] + df['low']) / 2) * 0.178404) + (df['vwap'] * (1 - 0.178404))

            # 计算 weighted_mid_vwap 的 3 天变化 (delta)
            df['delta_weighted_mid_vwap'] = df['weighted_mid_vwap'].diff(3)

            # 对 delta_weighted_mid_vwap 进行排名
            df['rank_delta_weighted_mid_vwap'] = df['delta_weighted_mid_vwap'].rank()

            # 计算最终因子 (rank_correlation - rank_delta_weighted_mid_vwap) 作为因子值
            df['factor'] =-1*(df['rank_correlation'] / df['rank_delta_weighted_mid_vwap'])

            return df

        # 计算所有股票的因子
        def calculate_all_factors(price_path):
            # 获取所有股票代码
            all_data_file = os.listdir(price_path)
            # 创建一个空的DataFrame来保存所有股票的因子
            all_factors = pd.DataFrame()

            # 遍历每一个股票代码，计算其因子，并添加到all_factors中
            for code in all_data_file:
                try:
                    df = pd.read_csv(price_path + '\\' + code)
                    # 对每一个数据都计算第一步定义的算法
                    factor = calculate_factor(df)
                    all_factors = all_factors.append(factor)
                except Exception as e:
                    print(code + '出现错误跳过:', e)

            return all_factors

        price_path = r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
        # 计算所有股票的因子，并且合并在一个表中
        all_factors = calculate_all_factors(price_path)

        all_factors = all_factors.dropna()

        # 对因子进行去除异常值和标准化操作
        all_factors = self.QuantileTransformer_and_standardize(all_factors)
        
        # 将结果保存到CSV文件中
        filename = 'PVCMRF'
        
        all_factors.reset_index(drop=True, inplace=True)

        # 使用self的holding_period和start_date参数
        result = self.add_holding_period_factors(all_factors, holding_period, start_date)

        result.to_csv(self.factor_export_path + "\\" + filename + '.csv', index=False)

        return result
    
    #量价关系类
    def PCVCM(self):
        # 这里我们直接使用类的属性，而不是在参数列表中使用self
        holding_period = self.holding_period
        
        start_date = self.start_date

        # 计算因子的函数
        def calculate_factor(df):
            # 确保数据按照时间排序
            df = df.sort_values('time')
            
            # 计算 adv30，即30天的平均成交量
            adv_window = 30
            df['adv30'] = df['volume'].rolling(window=adv_window).mean()
            
            # 计算 (high * 0.876703 + close * (1 - 0.876703))
            high_weight = 0.876703
            df['price_weighted'] = df['high'] * high_weight + df['close'] * (1 - high_weight)
            
            # 计算第一个相关性：price_weighted 与 adv30 的 9.61331 天相关性，并进行排名
            corr_window_1 = int(9.61331)  # 窗口期四舍五入为 9 天
            df['corr1'] = df['price_weighted'].rolling(window=corr_window_1).corr(df['adv30'])
            df['rank_corr1'] = df['corr1'].rank(pct=True)  # 用百分比排名
            
            # 计算 Ts_Rank((high + low) / 2, 3.70596) 和 Ts_Rank(volume, 10.1595)
            ts_rank_window_1 = int(3.70596)  # 窗口期四舍五入为 3 天
            ts_rank_window_2 = int(10.1595)  # 窗口期四舍五入为 10 天
            
            df['mid_price'] = (df['high'] + df['low']) / 2
            df['ts_rank_mid_price'] = df['mid_price'].rolling(window=ts_rank_window_1).apply(lambda x: x.rank().iloc[-1], raw=False)
            df['ts_rank_volume'] = df['volume'].rolling(window=ts_rank_window_2).apply(lambda x: x.rank().iloc[-1], raw=False)
            
            # 计算第二个相关性：ts_rank_mid_price 与 ts_rank_volume 的 7.11408 天相关性，并进行排名
            corr_window_2 = int(7.11408)  # 窗口期四舍五入为 7 天
            df['corr2'] = df['ts_rank_mid_price'].rolling(window=corr_window_2).corr(df['ts_rank_volume'])
            df['rank_corr2'] = df['corr2'].rank(pct=True)  # 用百分比排名
            
            # 计算因子值，使用 rank_corr1 和 rank_corr2 的幂运算
            df['factor'] = df['rank_corr1'] ** df['rank_corr2']
            
            return df

        # 计算所有股票的因子
        def calculate_all_factors(price_path):
            # 获取所有股票代码
            all_data_file = os.listdir(price_path)
            # 创建一个空的DataFrame来保存所有股票的因子
            all_factors = pd.DataFrame()

            # 遍历每一个股票代码，计算其因子，并添加到all_factors中
            for code in all_data_file:
                try:
                    df = pd.read_csv(price_path + '\\' + code)
                    # 对每一个数据都计算第一步定义的算法
                    factor = calculate_factor(df)
                    all_factors = all_factors.append(factor)
                except Exception as e:
                    print(code + '出现错误跳过:', e)

            return all_factors

        price_path = r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
        # 计算所有股票的因子，并且合并在一个表中
        all_factors = calculate_all_factors(price_path)

        all_factors = all_factors.dropna()

        # 对因子进行去除异常值和标准化操作
        all_factors = self.QuantileTransformer_and_standardize(all_factors)
        
        # 将结果保存到CSV文件中
        filename = 'PCVCM'
        
        all_factors.reset_index(drop=True, inplace=True)

        # 使用self的holding_period和start_date参数
        result = self.add_holding_period_factors(all_factors, holding_period, start_date)

        result.to_csv(self.factor_export_path + "\\" + filename + '.csv', index=False)

        return result
    
    def MRPD(self):
        # 这里我们直接使用类的属性，而不是在参数列表中使用self
        holding_period = self.holding_period
        
        start_date = self.start_date

        # 计算因子的函数
        def calculate_factor(df):
            # 确保数据按照时间排序
            df = df.sort_values('time')
            
            # 1. 计算 VWAP（成交量加权平均价）
            # VWAP = 累积的(价格 × 成交量) / 累积的成交量
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # 2. 计算 adv20（20 天平均成交量）
            adv_window = 20
            df['adv20'] = df['volume'].rolling(window=adv_window).mean()
            
            # 3. 计算过去 14 天的成交量累积和
            adv_sum_window = 14  # 14.7444 天简化为 14 天
            df['adv20_sum'] = df['adv20'].rolling(window=adv_sum_window).sum()
            
            # 4. 计算 close 与 adv20_sum 的 6 天相关性
            corr_window = 6  # 6.00049 天简化为 6 天
            df['corr_close_adv'] = df['close'].rolling(window=corr_window).corr(df['adv20_sum'])
            
            # 5. 对相关性进行 20 天时间序列排名
            ts_rank_window = 20  # 20.4195 天简化为 20 天
            df['ts_rank_corr'] = df['corr_close_adv'].rolling(window=ts_rank_window).apply(lambda x: x.rank().iloc[-1], raw=False)
            
            # 6. 计算 ((open + close) - (vwap + open))
            df['price_diff'] = (df['open'] + df['close']) - (df['vwap'] + df['open'])
            
            # 7. 对 price_diff 进行横截面排名
            df['rank_price_diff'] = df['price_diff'].rank(pct=True)
            
            # 比较 ts_rank_corr 和 rank_price_diff 并乘以 -1
            df['factor'] = (df['ts_rank_corr'] - df['rank_price_diff'])*-1
            
            return df

        # 计算所有股票的因子
        def calculate_all_factors(price_path):
            # 获取所有股票代码
            all_data_file = os.listdir(price_path)
            # 创建一个空的DataFrame来保存所有股票的因子
            all_factors = pd.DataFrame()

            # 遍历每一个股票代码，计算其因子，并添加到all_factors中
            for code in all_data_file:
                try:
                    df = pd.read_csv(price_path + '\\' + code)
                    # 对每一个数据都计算第一步定义的算法
                    factor = calculate_factor(df)
                    all_factors = all_factors.append(factor)
                except Exception as e:
                    print(code + '出现错误跳过:', e)

            return all_factors

        price_path = r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
        # 计算所有股票的因子，并且合并在一个表中
        all_factors = calculate_all_factors(price_path)

        all_factors = all_factors.dropna()

        # 对因子进行去除异常值和标准化操作
        all_factors = self.QuantileTransformer_and_standardize(all_factors)
        
        # 将结果保存到CSV文件中
        filename = 'MRPD'
        
        all_factors.reset_index(drop=True, inplace=True)

        # 使用self的holding_period和start_date参数
        result = self.add_holding_period_factors(all_factors, holding_period, start_date)

        result.to_csv(self.factor_export_path + "\\" + filename + '.csv', index=False)

        return result
    
    #一致预期类
    def Fore_Roe_Two_Ma_Factor(self,window_1=20,window_2=5):
        
        holding_period=self.holding_period
        
        start_date=self.start_date
        
        # 计算因子的函数
        def calculate_factor(df,window_1,window_2):
            
            df=df.sort_values(by='time')

            df['long_ma']=df['ths_fore_roe_mean_index'].rolling(window_1).mean()

            df['short_ma']=df['ths_fore_roe_mean_index'].rolling(window_2).mean()

            df['factor']=(df['short_ma']-df['long_ma'])/df['long_ma']

            return df

        # 计算所有股票的因子
        def calculate_all_factors(price_path,window_1,window_2):
            # 获取所有股票代码
            all_data_file = os.listdir(price_path)
            # 创建一个空的DataFrame来保存所有股票的因子
            all_factors = pd.DataFrame()

            # 遍历每一个股票代码，计算其因子，并添加到all_factors中
            for code in tqdm(all_data_file):
                try:
                    df=pd.read_csv(price_path+'\\'+code)
                    #对每一个数据都计算第一步定义的算法
                    factor = calculate_factor(df,window_1,window_2)
                    all_factors = all_factors.append(factor)
                except:
                    print(code+'出现错误跳过')
            

            return all_factors

        # 去除异常值并进行标准化

        price_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业一致预期数据'

        all_factors = calculate_all_factors(price_path,window_1,window_2)

        all_factors=all_factors.dropna()

            # 对因子进行去除异常值和标准化操作
        all_factors =self.winsorize_and_standardize(all_factors)

        all_factors.reset_index(drop=True,inplace=True)

        filename='fore_roe_two_ma_factor'+str(window_1)+'D_'+str(window_2)+'D'

        result=self.add_holding_period_factors(all_factors,holding_period,start_date)
        
        result.to_csv(self.factor_export_path+"\\"+filename+'.csv',index=False)

        return result

    #换手率类
    def Turn_Rate_Average(self,window_1=5):
        holding_period=self.holding_period
        start_date=self.start_date
        
        # 计算因子的函数
        def calculate_factor(df,window_1):
            
            df=df.sort_values(by='time')

            # 计算滚动平均换手率
            df['rolling_mean_window1'] = df['ths_free_turnover_ratio_index'].rolling(window=window_1).mean()
                # 计算因子值
            df['factor'] =-1* df['rolling_mean_window1']

            return df

        # 计算所有股票的因子
        def calculate_all_factors(price_path,window_1=5):
            # 获取所有股票代码
            all_data_file = os.listdir(price_path)
            # 创建一个空的DataFrame来保存所有股票的因子
            all_factors = pd.DataFrame()

            # 遍历每一个股票代码，计算其因子，并添加到all_factors中
            for code in all_data_file:
                try:
                    df=pd.read_csv(price_path+'\\'+code)
                    #对每一个数据都计算第一步定义的算法
                    factor = calculate_factor(df,window_1)
                    all_factors = all_factors.append(factor)
                except:
                    print(code+'出现错误跳过')
            

            return all_factors

                # 去除异常值并进行标准化

        price_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业自由流通换手率'

        all_factors = calculate_all_factors(price_path,window_1)

        all_factors=all_factors.dropna()

        # 对因子进行去除异常值和标准化操作
        all_factors =self.winsorize_and_standardize(all_factors)

        all_factors.reset_index(drop=True,inplace=True)

        filename='Turn_Rate_Average'+str(window_1)+'D'

        result=self.add_holding_period_factors(all_factors,holding_period,start_date)
        
        result.to_csv(self.factor_export_path+"\\"+filename+'.csv',index=False)

        return result

    def Free_Turn_Convergence_Factor_PCF(self):
                
        holding_period=self.holding_period
        start_date=self.start_date
        # 计算价格收敛因子（PCF）的函数
        def calculate_factor(df):
            # 确保数据按照时间排序
            df = df.sort_values('time')

            # ---- 计算不同周期的移动均线 ---- #
            df['ma1'] = df['ths_free_turnover_ratio_index']  # 当日收盘价即为MA1
            df['ma5'] = df['ths_free_turnover_ratio_index'].rolling(window=5).mean()
            df['ma10'] = df['ths_free_turnover_ratio_index'].rolling(window=10).mean()
            df['ma20'] = df['ths_free_turnover_ratio_index'].rolling(window=20).mean()
            df['ma60'] = df['ths_free_turnover_ratio_index'].rolling(window=60).mean()
            df['ma120'] = df['ths_free_turnover_ratio_index'].rolling(window=120).mean()

            # ---- 计算均线的标准差 ---- #
            # 注意：我们在计算标准差时需要确保所有的均线都有值，所以要处理缺失值
            df['std_ma'] = df[['ma1', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120']].std(axis=1)

            # ---- 计算价格收敛因子 ---- #
            # 使用 ∆log(1 + 标准差) 公式
            df['factor'] = -np.log(1 + df['std_ma'])

            return df

        # 计算所有股票的因子
        def calculate_all_factors(price_path):
            # 获取所有股票代码
            all_data_file = os.listdir(price_path)
            # 创建一个空的DataFrame来保存所有股票的因子
            all_factors = pd.DataFrame()

            # 遍历每一个股票代码，计算其因子，并添加到all_factors中
            for code in tqdm(all_data_file):
                try:
                    df = pd.read_csv(price_path + '\\' + code)
                    # 计算每一个股票的因子
                    factor = calculate_factor(df)
                    all_factors = all_factors.append(factor)
                except:
                    print(code + '出现错误跳过')

            return all_factors

            price_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业自由流通换手率'

        price_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业自由流通换手率'

        all_factors = calculate_all_factors(price_path)

        all_factors=all_factors.dropna()
        # 对因子进行去除异常值和标准化操作
        all_factors =self.winsorize_and_standardize(all_factors)

        all_factors.reset_index(drop=True,inplace=True)

        filename='Free_Turn_Convergence_Factor_PCF'

        result=self.add_holding_period_factors(all_factors,holding_period,start_date)
        
        result.to_csv(self.factor_export_path+"\\"+filename+'.csv',index=False)

        return result

    #资金流类
    def DDE_5D_factor(self):

        holding_period=self.holding_period

        start_date=self.start_date

        # 计算因子的函数
        def calculate_factor(df):

            df=df.sort_values(by='time')

            df.loc[:,'factor']=df.loc[:,'ths_dde_5d_hb_index']

            return df

        # 计算所有股票的因子
        def calculate_all_factors(price_path):
            # 获取所有股票代码
            all_data_file = os.listdir(price_path)
            # 创建一个空的DataFrame来保存所有股票的因子
            all_factors = pd.DataFrame()

            # 遍历每一个股票代码，计算其因子，并添加到all_factors中
            for code in all_data_file:
                try:
                    df=pd.read_csv(price_path+'\\'+code)
                    #对每一个数据都计算第一步定义的算法
                    factor = calculate_factor(df)
                    all_factors = all_factors.append(factor)
                except:
                    print(code+'出现错误跳过')

            return all_factors

        # 设置计算因子的数据路径
        price_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业资金流数据'
        # 计算所有股票的因子，并且合并在一个表中
        all_factors = calculate_all_factors(price_path)
        all_factors = all_factors.dropna()

        # 对因子进行去除异常值和标准化操作
        all_factors = self.winsorize_and_standardize(all_factors)
        all_factors.reset_index(drop=True,inplace=True)


        # 将结果保存到CSV文件中
        filename='DDE'+str(5)+'D'

        result=self.add_holding_period_factors(all_factors,holding_period,start_date)
        
        result.to_csv(self.factor_export_path+"\\"+filename+'.csv',index=False)
        
        return result


IS10=Industry_Strategy_10Day()

#动量类
Rank_Vwap=IS10.Rank_Vwap_Factor()

#反转类
LVVC=IS10.LVVC()

VRRF=IS10.VRRF()

PVCMRF=IS10.PVCMRF()

#量价关系类
PCVCM=IS10.PCVCM()

MRPD=IS10.MRPD()

#一致预期类
Fore_Roe_Two_Ma_Factor=IS10.Fore_Roe_Two_Ma_Factor()

#换手率类

Turn_Rate_Average=IS10.Turn_Rate_Average()

Free_Turn_Convergence_Factor_PCF=IS10.Free_Turn_Convergence_Factor_PCF()

#资金流
DDE_5D=IS10.DDE_5D_factor()




 #因子组合
def Portfolio(End_Date='2024-11-21'):

    #先组合相关性高的函数
    #换手率类

    factor_list=[Turn_Rate_Average,Free_Turn_Convergence_Factor_PCF]

    factor_name=["Turn_Rate_Average","Free_Turn_Convergence_Factor_PCF"]
    
    def compound_factor(factor_list,factor_name,End_Date):
        factors=[]

        for i,j in zip(factor_list,factor_name):

            data=i.set_index('time',drop=True)

            factor=data.loc[End_Date,:]

            factor=factor.set_index('thscode',drop=True)

            real_factor=factor[['real_standard_factor']]

            real_factor.columns=[j]

            factors.append(real_factor)
        
        factors=pd.concat(factors,axis=1)

        factors['factor'] = factors.mean(axis=1)

        factors['factor']=zscore(factors['factor'])

        result=factors[['factor']]

        result.columns=['real_standard_factor']
        
        return result

    Turn_Rate_factor=compound_factor(factor_list,factor_name,End_Date)
   
    #总组合    
    Factor_List=[Turn_Rate_factor,Rank_Vwap,LVVC,VRRF,PVCMRF,
                MRPD,Fore_Roe_Two_Ma_Factor,DDE_5D]
    
    Factor_Name=['Turn_Rate_factor', 'Rank_Vwap','LVVC','VRRF','PVCMRF','MRPD','Fore_Roe_Two_Ma_Factor',
                    'DDE_5D']
    
    Result=[]

    for factor, name in zip(Factor_List, Factor_Name):
        try:
            # 尝试设置'time'为索引
            factor = factor.set_index('time', drop=True)
            # 按日期筛选数据
            factor = factor.loc[End_Date, :]
            # 选择相关列
            factor = factor[['thscode', 'real_standard_factor']]
            # 将'代码'设置为索引
            factor.index = factor['thscode']
            # 仅保留标准因子列
            factor = factor[['real_standard_factor']]
        except KeyError:
            print(f"'time' column not found in DataFrame for {name}. Appending original DataFrame.")
            # 如果没有'time'列，继续使用原始DataFrame
        except Exception as e:
            print(f"An error occurred with {name}: {e}")
            continue
        
        # 确保列名在所有情况下都被设置
        factor.columns = [name]
        # 将处理后的数据添加到结果列表
        Result.append(factor)

    Result = pd.concat(Result, axis=1)

    Result['total_sum']=Result[Result.columns].sum(axis=1)

    Result=Result.sort_values(by='total_sum',ascending=False)

    Result=Result.dropna()

    export_path=r'D:\量化交易构建\ETF轮动策略\result'

    Result.to_excel(export_path+"\\"+'行业多因子结果'+'.xlsx')

    return Result

Port=Portfolio()


#筛出ETF组合
class select_invest_target():
    def __init__(self):

        self.financial_path=r'D:/量化交易构建/市场数据库/数据库/个股财务数据库'

    def Select_Industry_and_Build_Portfilo(self):

        Dirc=pd.read_excel(r'D:\量化交易构建\ETF轮动策略\result'+"\行业多因子结果.xlsx",index_col=[0])

        #选出前10

        Top_10=Dirc.iloc[:12,:]

        #选出对应的ETF代码和指数

        Top_10_Code=Top_10.index.to_list()

        #对应ETF名称文档

        Industary_Name=pd.read_excel(r'D:\量化交易构建\ETF轮动策略\result'+'\\申万二级行业代码.xlsx')

        df_filtered = Industary_Name[Industary_Name['代码'].isin(Top_10_Code)]

        df_filtered['代码'] = pd.Categorical(df_filtered['代码'], categories=Top_10_Code, ordered=True)

        df_filtered = df_filtered.sort_values('代码')
        
        df_filtered.to_excel(r'D:\量化交易构建\ETF轮动策略\result'+'\\'+'top10行业.xlsx')
        
        return df_filtered

    def Caculate_Port_Corr(self,index_code_list,window):
        
        path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'

        port=[]
        for i in index_code_list:

            data=pd.read_csv(path+'\\'+i+'.csv',index_col=[0])
            data.index=pd.to_datetime(data.index)
            close=data[['close']]
            close.columns=[i]
            port.append(close)
        
        port=pd.concat(port,axis=1)
        port=port.dropna()
        port=port.iloc[-window:,:]
        port_pct=port.pct_change()
        corr=port_pct.corr()

        return corr

    def thslogindemo(self):
        # 输入用户的帐号和密码
        thsLogin = THS_iFinDLogin("hwqh101","95df1a")
        print(thsLogin)
        if thsLogin != 0:
            print('登录失败')
        else:
            print('登录成功')

    def select_index_A_shares(self,index_code,date='20241115'):
        self.thslogindemo()
        df=THS_DR('p03563','date='+date+';'+'index_name='+index_code,'p03563_f001:Y,p03563_f002:Y,p03563_f003:Y,p03563_f004:Y','format:dataframe').data
        df.columns=['日期','代码','名称','权重']
        top_10=df.iloc[:12,:]
        code=top_10.loc[:,"代码"].tolist()
        
        return df,code
    
    def calculate_f_score(self,codes, date='2024-11-01'):
        """
        根据输入的股票代码列表和日期，计算Piotroski F-score，并返回结果DataFrame以及缺失的股票代码列表。
        
        参数:
        - codes: 股票代码列表 (如 ['600570.SH', '000001.SZ'])
        - date: 选择的日期 (如 '2022/1/14')
        - data_path: 数据存储路径 (如 'D:/量化交易构建/市场数据库/数据库/个股财务数据库')

        返回:
        - result_df: 包含股票代码和计算出的F-score的DataFrame
        - missing_codes: 缺失的股票代码列表
        """
        data_path=self.financial_path
        f_score_list = []  # 用于存储每只股票的F-score
        missing_codes = []  # 用于存储那些文件不存在的股票代码
        
        # 将输入的日期转换为datetime格式
        date = pd.to_datetime(date)
        
        # 遍历每只股票代码
        for code in codes:
            file_path = os.path.join(data_path, f"{code}.csv")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"文件 {file_path} 不存在，跳过该股票。")
                missing_codes.append(code)  # 将缺失的股票代码添加到列表中
                continue
            
            # 读取股票数据
            df = pd.read_csv(file_path)
            
            # 将time列转换为datetime格式，并设置为index
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # 按日期先排序
            df = df.sort_index(ascending=True)
            
            # 找到当前日期的数据
            if date not in df.index:
                print(f"股票 {code} 在日期 {date} 没有数据，跳过该股票。")
                continue
            
            df_current = df.loc[date]
            
            # 获取当前日期的索引
            current_index = df.index.get_loc(date)
            
            # 检查过去第252条数据是否存在
            if current_index < 252:
                print(f"股票 {code} 在过去252个交易日内没有足够的数据，跳过该股票。")
                continue
            
            # 取得过去第252条数据（前一年左右的数据）
            df_previous = df.iloc[current_index - 252]
            
            # 获取当前和前一年的财务数据
            net_income = df_current['ths_np_atoopc_pit_stock']  # 归属于母公司所有者的净利润
            total_assets = df_current['ths_total_assets_pit_stock']  # 总资产
            cash_flow = df_current['ths_ncf_from_oa_pit_stock']  # 经营活动产生的现金流量净额
            total_liab = df_current['ths_total_liab_pit_stock']  # 总负债
            current_assets = df_current['ths_total_current_assets_pit_stock']  # 流动资产
            current_liab = df_current['ths_total_current_liab_pit_stock']  # 流动负债
            shares = df_current['ths_total_shares_stock']  # 总股本
            revenue = df_current['ths_operating_total_revenue_stock']  # 营业总收入
            gross_margin = df_current['ths_gross_selling_rate_stock']  # 销售毛利率
            
            net_income_prev = df_previous['ths_np_atoopc_pit_stock']
            total_assets_prev = df_previous['ths_total_assets_pit_stock']
            total_liab_prev = df_previous['ths_total_liab_pit_stock']
            current_assets_prev = df_previous['ths_total_current_assets_pit_stock']
            current_liab_prev = df_previous['ths_total_current_liab_pit_stock']
            revenue_prev = df_previous['ths_operating_total_revenue_stock']
            gross_margin_prev = df_previous['ths_gross_selling_rate_stock']

            # 初始化F-score为0
            f_score = 0
            
            # 盈利能力 (4项)
            ROA = net_income / total_assets  # 计算ROA
            ROA_prev = net_income_prev / total_assets_prev  # 前一年ROA
            if net_income > 0:
                f_score += 1  # 净利润为正
            if cash_flow > 0:
                f_score += 1  # 经营现金流为正
            if ROA > ROA_prev:
                f_score += 1  # 资产回报率较前一年提高
            if cash_flow > net_income:
                f_score += 1  # 经营现金流大于净利润
            
            # 杠杆、流动性和融资状况 (3项)
            if total_liab / total_assets < total_liab_prev / total_assets_prev:
                f_score += 1  # 杠杆比率较前一年下降
            current_ratio = current_assets / current_liab
            current_ratio_prev = current_assets_prev / current_liab_prev
            if current_ratio > current_ratio_prev:
                f_score += 1  # 流动比率较前一年提高
            if shares == df_previous['ths_total_shares_stock']:
                f_score += 1  # 没有发新股
            
            # 经营效率 (2项)
            if gross_margin > gross_margin_prev:
                f_score += 1  # 毛利率较前一年提高
            asset_turnover = revenue / total_assets
            asset_turnover_prev = revenue_prev / total_assets_prev
            if asset_turnover > asset_turnover_prev:
                f_score += 1  # 资产周转率较前一年提高
            
            # 将结果存储到列表
            f_score_list.append({
                'code': code,
                'F-score': f_score
            })
        
        # 将结果转换为DataFrame
        result_df = pd.DataFrame(f_score_list)
        
        return result_df, missing_codes  # 返回结果以及缺失的股票代码

    def selecting_Final_A_Share_Port(self,indsutry_codes):
        result=[]

        missing_codes_list=[]
        for i in indsutry_codes:

            try:

                df,codes=self.select_index_A_shares(i)

                f_score_df,missing_codes=self.calculate_f_score(codes)

                df=df.set_index('代码',drop=True)

                f_score_df=f_score_df.set_index('code',drop=True)

                merge_df=pd.merge(df,f_score_df,right_index=True,left_index=True)

                result.append(merge_df)

                missing_codes_list.append(missing_codes)
            except:

                print(i+'出现错误')

        result=pd.concat(result,axis=0)

        missing_codes_list=list(itertools.chain(*missing_codes_list))

        return result,missing_codes_list

sit=select_invest_target()
etf_result=sit.Select_Industry_and_Build_Portfilo()
code_list=etf_result.loc[:,"代码"].to_list()
Corr=sit.Caculate_Port_Corr(code_list,90)


#剔除相关性
selected_codes = []
for column in Corr.columns:
    if all(Corr.loc[column, selected_code] < 0.99 for selected_code in selected_codes):
        selected_codes.append(column)

# 输出筛选后的代码列表
print(selected_codes)
etf=etf_result[etf_result['代码'].isin(selected_codes)]

etf_codes=etf.loc[:,"代码"].tolist()

df,missing_codes=sit.selecting_Final_A_Share_Port(etf_codes)

df.to_excel(r'D:\量化交易构建\ETF轮动策略\行业轮动策略\result'+"\\"+'个股结果.xlsx')