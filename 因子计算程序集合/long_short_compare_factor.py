# 导入需要的模块
import os
import pandas as pd
import numpy as np
from scipy.stats import mstats, zscore
from tqdm import tqdm

# 计算因子的函数
def calculate_factor(df,window_1,window_2):

    # 确保数据按照时间排序
    df = df.sort_values('time')
        # 定义函数计算因子值
        #因子定义因子的定义为<首先计算多空力量对比。分子我们用多头力量与空头力量相减，即
        # (𝐶𝑙𝑜𝑠𝑒 − 𝐿𝑜𝑤) − (𝐻𝑖𝑔ℎ − 𝐶𝑙𝑜𝑠𝑒)，分母为最高价减去最低价，也就是日内价格区
        # 间的极值。再将所得多空力量对比乘上当日行业成交量，可得当日多空力量对比
        # 的金额绝对值。我们用长期每日多空力量对比的指数加权平均值，减去短期每日多空力量对比的
        # 指数加权平均值，可以得到近期多空力量对比相对于长期多空力量对比均值的变化。
        # 因子值越大，说明近期多头相对于空头力量减弱；因子值越小，说明近期多空对
        # 比度相对于长期加大
    df['LS_Compare'] = df['volume'] * ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    long_term = df['LS_Compare'].ewm(span=window_1).mean()
    short_term = df['LS_Compare'].ewm(span=window_2).mean()
    df['factor'] = -(long_term - short_term)

    return df

# 计算所有股票的因子
def calculate_all_factors(price_path,window_1=25,window_2=10):
    # 获取所有股票代码
    all_data_file = os.listdir(price_path)
    # 创建一个空的DataFrame来保存所有股票的因子
    all_factors = pd.DataFrame()

    # 遍历每一个股票代码，计算其因子，并添加到all_factors中
    for code in all_data_file:
        try:
            df=pd.read_csv(price_path+'\\'+code)
            #对每一个数据都计算第一步定义的算法
            factor = calculate_factor(df,window_1,window_2)
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
    filename = 'long_short_compare_factor' + str(25) + 'D_' + str(10) + 'D'
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库" + "\\" + filename + '.csv', index=False)


# 如果这个脚本被作为主程序运行，则运行main()函数
if __name__ == "__main__":
    main()