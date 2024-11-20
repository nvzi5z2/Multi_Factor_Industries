
import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata,zscore
from tqdm import tqdm

# 计算因子的函数
def calculate_factor(df):
    # 确保数据按照时间排序
    df = df.sort_values('time')

    # ---- 计算 Alpha#20 因子 ---- #

    # 计算 open - delay(high, 1)
    df['open_high_diff'] = df['open'] - df['high'].shift(1)

    # 计算 open - delay(close, 1)
    df['open_close_diff'] = df['open'] - df['close'].shift(1)

    # 计算 open - delay(low, 1)
    df['open_low_diff'] = df['open'] - df['low'].shift(1)

    # 计算排名 rank(open - delay(high, 1))
    df['rank_open_high'] = rankdata(df['open_high_diff'])

    # 计算排名 rank(open - delay(close, 1))
    df['rank_open_close'] = rankdata(df['open_close_diff'])

    # 计算排名 rank(open - delay(low, 1))
    df['rank_open_low'] = rankdata(df['open_low_diff'])

    # 计算最终的因子值
    df['factor'] = ( df['rank_open_high']) * df['rank_open_close'] * df['rank_open_low']

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

# 去除异常值并进行标准化
def winsorize_and_standardize(df):
    # 使用标准差方法处理极端值
    def limit_to_std(x):
        mean = x.mean()
        std = x.std()
        x = x.apply(lambda v: min(max(v, mean - 3 * std), mean + 3 * std))
        return x

    # 应用limit_to_std函数处理每个时间组的因子值
    df['standard_factor'] = df['factor'].groupby(df['time']).transform(limit_to_std)

    # 使用zscore方法进行标准化
    df['standard_factor'] = df.groupby('time')['standard_factor'].transform(zscore)

    result = df[['time', 'thscode', 'factor', 'standard_factor']]

    return result

# 主程序
def main():
    # 设置计算因子的数据路径
    price_path = r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
    # 计算所有股票的因子，并且合并在一个表中
    all_factors = calculate_all_factors(price_path)
    all_factors = all_factors.dropna()

    # 对因子进行去除异常值和标准化操作
    all_factors = winsorize_and_standardize(all_factors)

    # 将结果保存到CSV文件中
    filename = 'Open_Relative_to_Previous_Day_Reversal'
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库" + "\\" + filename + '.csv', index=False)

# 如果这个脚本被作为主程序运行，则运行main()函数
if __name__ == "__main__":
    main()