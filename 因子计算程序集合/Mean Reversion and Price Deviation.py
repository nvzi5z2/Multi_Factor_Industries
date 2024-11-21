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
    filename='MRPD'
    
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\待测试因子库"+"\\"+filename+'.csv',index=False)

    result=all_factors.set_index('time',drop=True)

    result=result.loc["2024-09-11",:]

    result[['factor']].plot.hist(bins=20)

    result[['standard_factor']].plot.hist(bins=20)

main()