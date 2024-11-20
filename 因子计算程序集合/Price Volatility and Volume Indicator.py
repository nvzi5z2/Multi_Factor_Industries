# 导入需要的模块
import os
import pandas as pd
import numpy as np
from scipy.stats import mstats, zscore
from scipy.stats import boxcox, zscore
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer

def calculate_factor(df):
    # 确保数据按照时间排序
    df = df.sort_values('time')

    # 计算 VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # 计算 (high - low) / (5日均收盘价)
    df['price_range'] = (df['high'] - df['low']) / df['close'].rolling(window=5).mean()

    # 计算延迟2天的 price_range
    df['price_range_delay'] = df['price_range'].shift(2)

    # 计算 rank(price_range_delay)
    df['rank_price_range_delay'] = df['price_range_delay'].rank()

    # 计算 rank(rank(volume))
    df['rank_volume'] = df['volume'].rank()
    df['rank_rank_volume'] = df['rank_volume'].rank()

    # 计算高低价格波动率 / VWAP 与收盘价的差值
    df['price_vwap_diff'] = df['vwap'] - df['close']
    
    # 防止除以零操作，添加一个小常数
    df['price_vwap_indicator'] = df['price_range'] / (df['price_vwap_diff'] + np.finfo(float).eps)

    # 计算最终因子
    df['factor'] = (df['rank_price_range_delay'] * df['rank_rank_volume']) / df['price_vwap_indicator']

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


# def winsorize_and_standardize(df):
#     # 填充缺失值
#     df['factor'] = df.groupby('thscode')['factor'].fillna(method='ffill').fillna(method='bfill')

#     # 极端值处理函数：将因子值限制在3倍标准差范围内，并处理小样本问题
#     def limit_to_std(x):
#         # 如果样本量小于5，跳过极端值处理
#         if len(x) < 5:
#             return x
        
#         mean = x.mean()
#         std = x.std()
#         # 将超过3倍标准差的值设为3倍标准差
#         x = x.apply(lambda v: min(max(v, mean - 3 * std), mean + 3 * std))
#         return x

#     # 应用极端值处理到每个时间段的因子值
#     df['standard_factor'] = df.groupby('time')['factor'].transform(limit_to_std)
    
#     # 使用安全的zscore方法进行标准化
#     df['standard_factor'] = df.groupby('time')['standard_factor'].transform(zscore)
    
#     result = df[['time', 'thscode', 'factor', 'standard_factor']]

#     return result


# Winsorization函数
# 检查并过滤掉 inf 和 -inf 值
def winsorize_and_standardize(df):
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


def main():

    #设置计算因子的数据路径
    price_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
    # 计算所有股票的因子,并且合并在一个表中
    all_factors = calculate_all_factors(price_path)
    all_factors=all_factors.dropna()

    # 对因子进行去除异常值和标准化操作
    all_factors = winsorize_and_standardize(all_factors)

    # 将结果保存到CSV文件中
    filename='PVVI'
    
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库"+"\\"+filename+'.csv',index=False)

    result=all_factors.set_index('time',drop=True)

    result=result.loc["2024-09-11",:]

    result[['factor']].plot.hist(bins=20)

    result[['standard_factor']].plot.hist(bins=20)

main()