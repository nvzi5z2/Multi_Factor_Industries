# 导入需要的模块
import os
import pandas as pd
import numpy as np
from scipy.stats import mstats, zscore
from tqdm import tqdm

# 计算因子的函数
def calculate_factor(df, window=250):

    # 确保数据按照时间排序
    df = df.sort_values('time')
    
    # 计算价格动量 delta(close, 1) 和前一天的价格动量 delta(delay(close, 1), 1)
    df['delta_close_1'] = df['close'].diff(1)
    df['delta_delay_close_1'] = df['close'].shift(1).diff(1)
    
    # 计算250天的相关性
    df['correlation'] = df['delta_close_1'].rolling(window=window).corr(df['delta_delay_close_1'])
    
    # 计算 (correlation * delta(close, 1)) / close
    df['part1'] = (df['correlation'] * df['delta_close_1']) / df['close']
    
    # 计算过去250天价格动量平方和
    df['momentum_squared'] = (df['delta_close_1'] / df['close'].shift(1)) ** 2
    df['sum_momentum_squared'] = df['momentum_squared'].rolling(window=window).sum()
    
    # 计算最终因子值
    df['factor'] = df['part1'] / df['sum_momentum_squared']
    
    return df[['time', 'thscode', 'factor']]
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


#主程序

def main():

    #设置计算因子的数据路径
    price_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
    # 计算所有股票的因子,并且合并在一个表中
    all_factors = calculate_all_factors(price_path)
    all_factors=all_factors.dropna()

    # 对因子进行去除异常值和标准化操作
    all_factors = winsorize_and_standardize(all_factors)

    # 将结果保存到CSV文件中
    filename='VWRF'
    
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库"+"\\"+filename+'.csv',index=False)

    result=all_factors.set_index('time',drop=True)

    result=result.loc["2024-11-01",:]

    result[['factor']].plot.hist(bins=30)

    result[['standard_factor']].plot.hist(bins=30)


main()