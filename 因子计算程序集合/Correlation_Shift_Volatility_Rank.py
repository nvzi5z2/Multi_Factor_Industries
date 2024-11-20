
import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata,zscore
from tqdm import tqdm


# 计算因子的函数
def calculate_factor(df, window_corr=5, window_vol=20, delta_window=5):
    # 确保数据按照时间排序

    def rolling_correlation(x, y, window):
        return x.rolling(window).corr(y)
        
    df = df.sort_values('time')

    # ---- 计算 Alpha#22 因子 ---- #

    # 计算高价和成交量的5日滚动相关性
    df['corr_high_vol'] = rolling_correlation(df['high'], df['volume'], window_corr)

    # 计算相关性5日变化量
    df['delta_corr'] = df['corr_high_vol'].diff(delta_window)

    # 计算20日收盘价标准差
    df['stddev_close_20'] = df['close'].rolling(window_vol).std()

    # 对标准差进行排名
    df['rank_stddev'] = rankdata(df['stddev_close_20'])

    # 计算最终的因子值
    df['factor'] = -1 * (df['delta_corr'] * df['rank_stddev'])

    return df

# 计算所有股票的因子
def calculate_all_factors(price_path, window_corr=5, window_vol=20, delta_window=5):
    # 获取所有股票代码
    all_data_file = os.listdir(price_path)
    # 创建一个空的DataFrame来保存所有股票的因子
    all_factors = pd.DataFrame()

    # 遍历每一个股票代码，计算其因子，并添加到all_factors中
    for code in tqdm(all_data_file):
        try:
            df = pd.read_csv(price_path + '\\' + code)
            # 计算每一个股票的因子
            factor = calculate_factor(df, window_corr, window_vol, delta_window)
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
    filename = 'Correlation_Shift_Volatility_Rank_5_20D'
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库" + "\\" + filename + '.csv', index=False)

# 如果这个脚本被作为主程序运行，则运行main()函数
if __name__ == "__main__":
    main()