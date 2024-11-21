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
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # 2. 计算加权的收盘价和 VWAP
    weight_close = 0.369701
    weight_vwap = 1 - weight_close
    df['weighted_price'] = df['close'] * weight_close + df['vwap'] * weight_vwap
    
    # 3. 计算 delta（变化率）
    delta_window = int(1.91233)  # 窗口期取 2 天
    df['delta_weighted_price'] = df['weighted_price'].diff(periods=delta_window)
    
    # 4. 对 delta 应用线性衰减
    decay_window_1 = int(2.65461)  # 窗口期取 3 天
    df['decay_delta'] = df['delta_weighted_price'].rolling(window=decay_window_1).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
    
    # 5. 对 decay_delta 进行横截面排名
    df['rank_decay_delta'] = df['decay_delta'].rank(pct=True)
    
    # 6. 计算 adv81（81 天平均成交量）
    adv_window = 81
    df['adv81'] = df['volume'].rolling(window=adv_window).mean()
    
    # 7. 计算 adv81 与 close 的相关性
    corr_window = int(13.4132)  # 窗口期取 13 天
    df['corr_adv_close'] = df['adv81'].rolling(window=corr_window).corr(df['close'])
    
    # 8. 取相关性的绝对值并应用线性衰减
    df['abs_corr_adv_close'] = df['corr_adv_close'].abs()
    decay_window_2 = int(4.89768)  # 窗口期取 5 天
    df['decay_corr'] = df['abs_corr_adv_close'].rolling(window=decay_window_2).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
    
    # 9. 对 decay_corr 进行时间序列排名
    ts_rank_window = int(14.4535)  # 窗口期取 14 天
    df['ts_rank_decay_corr'] = df['decay_corr'].rolling(window=ts_rank_window).apply(lambda x: x.rank().iloc[-1], raw=False)
    
    # 10. 取 rank_decay_delta 和 ts_rank_decay_corr 的最大值
    df['max_rank'] = np.maximum(df['rank_decay_delta'], df['ts_rank_decay_corr'])
    
    # 11. 因子取反
    df['factor'] = -df['max_rank']
    
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
    filename='PWCR'
    
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\待测试因子库"+"\\"+filename+'.csv',index=False)

    result=all_factors.set_index('time',drop=True)

    result=result.loc["2024-09-11",:]

    result[['factor']].plot.hist(bins=20)

    result[['standard_factor']].plot.hist(bins=20)

main()