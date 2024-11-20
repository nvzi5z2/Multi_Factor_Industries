# 导入需要的模块
import os
import pandas as pd
import numpy as np
from scipy.stats import mstats, zscore
from scipy.stats import boxcox, zscore
from tqdm import tqdm

# 
# 计算因子的函数
def calculate_factor(df, window=180):
    # 确保数据按照时间排序
    df = df.sort_values('time')

    # 计算成交量加权平均价（VWAP）
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # 计算过去16天VWAP的最小值
    df['vwap_min_16'] = df['vwap'].rolling(window=16).min()

    # 计算VWAP与其最小值的差异
    df['vwap_diff'] = df['vwap'] - df['vwap_min_16']

    # 对vwap_diff进行排名 (rank)
    df['rank_vwap_diff'] = df['vwap_diff'].rank()

    # 计算180天的平均成交量 (adv180)
    df['adv180'] = df['volume'].rolling(window=window).mean()

    # 计算VWAP与adv180的17天相关性
    df['correlation_vwap_adv180'] = df['vwap'].rolling(window=17).corr(df['adv180'])

    # 对correlation_vwap_adv180进行排名 (rank)
    df['rank_correlation'] = df['correlation_vwap_adv180'].rank(pct=True)

    # 计算最终因子 (rank_vwap_diff < rank_correlation)
    df['factor'] = df['rank_correlation'] - df['rank_price_momentum']

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
    filename='VWCLF'
    
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库"+"\\"+filename+'.csv',index=False)

    result=all_factors.set_index('time',drop=True)

    result=result.loc["2024-09-10",:]

    result[['factor']].plot.hist(bins=20)

    result[['standard_factor']].plot.hist(bins=20)

main()