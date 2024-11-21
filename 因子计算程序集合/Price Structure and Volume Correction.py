# 导入需要的模块
import os
import pandas as pd
import numpy as np
from scipy.stats import mstats, zscore
from scipy.stats import boxcox, zscore
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer

# 计算因子的函数
def calculate_factor(df):
    # 确保数据按照时间排序
    df = df.sort_values('time')
    
    # 第一部分：计算价格结构信号
    df['price_structure'] = (((df['high'] + df['low']) / 2) + df['close']) < (df['low'] + df['open'])
    decay_window_1 = int(14.7221)  # 窗口期约为 15 天
    df['decay_price_structure'] = df['price_structure'].rolling(window=decay_window_1).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    )
    ts_rank_window_1 = int(18.8683)  # 窗口期约为 19 天
    df['ts_rank_price_structure'] = df['decay_price_structure'].rolling(window=ts_rank_window_1).apply(
        lambda x: x.rank().iloc[-1] if len(x) == ts_rank_window_1 else np.nan, raw=False
    )
    
    # 第二部分：计算最低价与成交量的相关性信号
    adv_window = 30  # 窗口期为 30 天
    df['adv30'] = df['volume'].rolling(window=adv_window).mean()
    corr_window = int(7.58555)  # 窗口期约为 8 天
    df['corr_low_adv'] = df['low'].rank(pct=True).rolling(window=corr_window).corr(df['adv30'].rank(pct=True))
    decay_window_2 = int(6.94024)  # 窗口期约为 7 天
    df['decay_corr_low_adv'] = df['corr_low_adv'].rolling(window=decay_window_2).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    )
    ts_rank_window_2 = int(6.80584)  # 窗口期约为 7 天
    df['ts_rank_corr_low_adv'] = df['decay_corr_low_adv'].rolling(window=ts_rank_window_2).apply(
        lambda x: x.rank().iloc[-1] if len(x) == ts_rank_window_2 else np.nan, raw=False
    )
    
    df['factor'] =  df['ts_rank_corr_low_adv']-df['ts_rank_price_structure']-df['ts_rank_corr_low_adv']
    
    
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
# def winsorize_and_standardize(df):
#     # 使用标准差方法处理极端值
#     def limit_to_std(x):
#         # 计算平均值和标准差
#         mean = x.mean()
#         std = x.std()
#         # 将超过3倍标准差的值设为3倍标准差
#         x = x.apply(lambda v: min(max(v, mean - 3 * std), mean + 3 * std))
#         return x
    
#     # 应用limit_to_std函数处理每个时间组的因子值
#     df['standard_factor'] = df['factor'].groupby(df['time']).transform(limit_to_std)
    
#     # 使用zscore方法进行标准化
#     df['standard_factor'] = df.groupby('time')['standard_factor'].transform(zscore)
    
#     result = df[['time','thscode', 'factor', 'standard_factor']]

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
    filename='PSVC'
    
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\待测试因子库"+"\\"+filename+'.csv',index=False)

    result=all_factors.set_index('time',drop=True)

    result=result.loc["2024-09-11",:]

    result[['factor']].plot.hist(bins=20)

    result[['standard_factor']].plot.hist(bins=20)

main()