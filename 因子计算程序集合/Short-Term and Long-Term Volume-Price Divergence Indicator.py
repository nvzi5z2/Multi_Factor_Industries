# 导入需要的模块
import os
import pandas as pd
import numpy as np
from scipy.stats import mstats, zscore
from scipy.stats import boxcox, zscore
from tqdm import tqdm

# 计算因子的函数
def calculate_factor(df):

    # 确保数据按照时间排序
    df = df.sort_values('time')

    # 计算 VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # 计算ADV10 (10日平均成交量)
    df['adv10'] = df['volume'].rolling(window=10).mean()

    # 计算49日窗口期内 vwap 和 adv10 的相关性
    def rolling_corr(series1, series2, window):
        return series1.rolling(window).corr(series2)

    df['corr_vwap_adv10'] = rolling_corr(df['vwap'], df['adv10'].rolling(window=50).sum(), window=8)

    # 计算 rank(corr_vwap_adv10^4)
    df['rank_corr_vwap_adv10_4'] = df['corr_vwap_adv10'] ** 4
    df['rank_corr_vwap_adv10_4_rank'] = df['rank_corr_vwap_adv10_4'].rank()

    # 计算14日窗口期内的累乘值，并取对数
    df['product_rank_corr_vwap_adv10'] = df['rank_corr_vwap_adv10_4_rank'].rolling(window=15).apply(np.prod, raw=True)
    df['log_product_rank_corr_vwap_adv10'] = np.log(df['product_rank_corr_vwap_adv10'].replace(0, np.nan) + 1)  # +1 避免log(0)

    # 对 log_product_rank_corr_vwap_adv10 进行排名
    df['rank_log_product'] = df['log_product_rank_corr_vwap_adv10'].rank()

    # 计算 vwap 和 volume 的 rank 相关性 (5日窗口期)
    df['rank_vwap'] = df['vwap'].rank()
    df['rank_volume'] = df['volume'].rank()
    df['corr_vwap_volume'] = rolling_corr(df['rank_vwap'], df['rank_volume'], window=5)

    # 对 corr_vwap_volume 进行排名
    df['rank_corr_vwap_volume'] = df['corr_vwap_volume'].rank()

    # 计算最终因子：比较 rank_log_product 和 rank_corr_vwap_volume，结果乘以 -1
    df['factor'] = (df['rank_log_product'] -df['rank_corr_vwap_volume'])

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


# Winsorization函数
# def winsorize_and_standardize(df):
#     # Winsorization：将极端值限制在1%和99%的分位数之间
#     def winsorize_series(x):
#         lower_bound = x.quantile(0.01)
#         upper_bound = x.quantile(0.99)
#         return np.clip(x, lower_bound, upper_bound)

#     # 应用Winsorization到每个时间段的因子值
#     df['winsorized_factor'] = df.groupby('time')['factor'].transform(winsorize_series)
    
#     # 使用zscore方法进行标准化
#     df['standard_factor'] = df.groupby('time')['winsorized_factor'].transform(zscore)
    
#     result = df[['time', 'thscode', 'factor', 'standard_factor']]

#     return result

def main():

    #设置计算因子的数据路径
    price_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
    # 计算所有股票的因子,并且合并在一个表中
    all_factors = calculate_all_factors(price_path)
    all_factors=all_factors.dropna()

    # 对因子进行去除异常值和标准化操作
    all_factors = winsorize_and_standardize(all_factors)

    # 将结果保存到CSV文件中
    filename='STLVDI'
    
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库"+"\\"+filename+'.csv',index=False)

    result=all_factors.set_index('time',drop=True)

    result=result.loc["2024-09-11",:]

    result[['factor']].plot.hist(bins=20)

    result[['standard_factor']].plot.hist(bins=20)

main()