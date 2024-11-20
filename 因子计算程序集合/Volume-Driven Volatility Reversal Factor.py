# 导入需要的模块
import os
import pandas as pd
import numpy as np
from scipy.stats import mstats, zscore
from tqdm import tqdm


# 计算因子的函数
def calculate_factor(df):

    # 确保数据按照时间排序
    df = df.sort_values('time')

    # 计算 (close - open) 和前一天交易量的相关性
    df['price_change'] = df['close'] - df['open']
    df['volume_lag1'] = df['volume'].shift(1)
    df['corr_price_volume'] = df['price_change'].rolling(window=15).corr(df['volume_lag1'])
    df['corr_price_volume_rank'] = df['corr_price_volume'].rank(pct=True)

    # 计算 (open - close) 的排名
    df['open_close_diff'] = df['open'] - df['close']
    df['open_close_diff_rank'] = df['open_close_diff'].rank(pct=True)

    # 计算过去6天的负收益率并进行5天的时间序列排名
    df['returns'] = df['close'].pct_change()
    df['neg_returns_lag6'] = -df['returns'].shift(6)
    df['ts_rank_neg_returns'] = df['neg_returns_lag6'].rolling(window=5).apply(lambda x: x.rank(pct=True).iloc[-1])

    # 计算 VWAP 和 adv20 的6天相关性
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['adv20'] = df['volume'].rolling(window=20).mean()
    df['corr_vwap_adv20'] = df['vwap'].rolling(window=6).corr(df['adv20'])
    df['corr_vwap_adv20_rank'] = df['corr_vwap_adv20'].rank(pct=True)

    # 计算长期均线和开盘价、收盘价的差异
    df['close_200_avg'] = df['close'].rolling(window=200).mean()
    df['long_short_diff'] = (df['close_200_avg'] - df['open']) * (df['close'] - df['open'])
    df['long_short_diff_rank'] = df['long_short_diff'].rank(pct=True)

    # 计算最终因子值
    df['factor'] = (2.21 * df['corr_price_volume_rank']) + \
                   (0.7 * df['open_close_diff_rank']) + \
                   (0.73 * df['ts_rank_neg_returns']) + \
                   df['corr_vwap_adv20_rank'] + \
                   (0.6 * df['long_short_diff_rank'])

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
            df=pd.read_csv(price_path+'\\'+code)
            #对每一个数据都计算第一步定义的算法
            factor = calculate_factor(df)
            all_factors = all_factors.append(factor)
        except:
            print(code+'出现错误跳过')

    return all_factors

# 去除异常值并进行标准化
def winsorize_and_standardize(df):
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
    
    result = df[['time','thscode', 'factor', 'standard_factor']]

    return result
# 主程序

def main():

    #设置计算因子的数据路径
    price_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
    # 计算所有股票的因子,并且合并在一个表中
    all_factors = calculate_all_factors(price_path)
    all_factors=all_factors.dropna()

    # 对因子进行去除异常值和标准化操作
    all_factors = winsorize_and_standardize(all_factors)

    # 将结果保存到CSV文件中
    filename='VDVRF'+str(32)+'D'+str(16)+'D'+str(32)+'D'
    
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库"+"\\"+filename+'.csv',index=False)

    result=all_factors.set_index('time',drop=True)

    result=result.loc["2024-11-01",:]

    result[['factor']].plot.hist(bins=30)

    result[['standard_factor']].plot.hist(bins=30)


main()