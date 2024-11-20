
import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata,zscore
from tqdm import tqdm

# 计算因子的函数
def calculate_factor(df, window_short=5, window_long=10, adv_window=20):
    # 确保数据按照时间排序
    df = df.sort_values('time')

    # ---- 计算 Alpha#21 因子的修改版本 ---- #

    # 计算 2日收盘均值
    df['avg_close_2'] = df['close'].rolling(window_short).mean()

    # 计算 8日收盘均值
    df['avg_close_8'] = df['close'].rolling(window_long).mean()

    # 计算 8日收盘标准差
    df['stddev_close_8'] = df['close'].rolling(window_long).std()

    # 计算 20日平均成交量
    df['adv20'] = df['volume'].rolling(adv_window).mean()

    # 计算 volume / adv20
    df['volume_adv_ratio'] = df['volume'] / df['adv20']

    # 计算价格差异的强弱因子
    price_diff_1 = (df['avg_close_2'] - (df['avg_close_8'] + df['stddev_close_8']))
    price_diff_2 = (df['avg_close_2'] - (df['avg_close_8'] - df['stddev_close_8']))

    # 对价格差异进行标准化处理，得到强弱对比
    df['price_factor_1'] = zscore(price_diff_1.fillna(0))  # 标准化
    df['price_factor_2'] = zscore(price_diff_2.fillna(0))  # 标准化

    # 成交量与均量的比值作为强弱因子
    df['volume_factor'] = zscore(df['volume_adv_ratio'].fillna(0))

    # 综合价格和成交量的强弱因子，得到最终因子值
    df['factor'] = np.where(
        df['price_factor_1'] > 0,  # 如果价格因子1大于0
        -df['price_factor_1'],  # 越大代表越看空
        np.where(
            df['price_factor_2'] < 0,  # 如果价格因子2小于0
            df['price_factor_2'],  # 越小代表越看多
            df['volume_factor']  # 否则根据成交量因子判断
        )
    )

    return df

# 计算所有股票的因子
def calculate_all_factors(price_path, window_short=2, window_long=8, adv_window=20):
    # 获取所有股票代码
    all_data_file = os.listdir(price_path)
    # 创建一个空的DataFrame来保存所有股票的因子
    all_factors = pd.DataFrame()

    # 遍历每一个股票代码，计算其因子，并添加到all_factors中
    for code in tqdm(all_data_file):
        try:
            df = pd.read_csv(price_path + '\\' + code)
            # 计算每一个股票的因子
            factor = calculate_factor(df, window_short, window_long, adv_window)
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
    filename = 'Price_Deviation_Volume_Strength_5_10_20D'
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库" + "\\" + filename + '.csv', index=False)

# 如果这个脚本被作为主程序运行，则运行main()函数
if __name__ == "__main__":
    main()