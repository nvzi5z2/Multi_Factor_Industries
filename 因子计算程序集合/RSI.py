# 导入需要的模块
import os
import pandas as pd
import numpy as np
from scipy.stats import mstats, zscore
from tqdm import tqdm


# 计算因子的函数
def calculate_factor(df,window):
    """
    计算RSI截面因子的函数。

    参数:
    df : pandas.DataFrame
        包含至少包括'time'和'close'列的DataFrame，其中
        'time'是时间戳，'close'是每日的收盘价。
    window : int
        用于计算RSI的时间窗口长度。

    返回:
    df : pandas.DataFrame
        增加了'RSI'列的DataFrame。
    """
    # 确保数据按照时间排序
    df = df.sort_values('time')
    
    # 计算每日的价格变化
    delta = df['close'].diff()

    # 分离涨跌变化
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # 计算平均涨跌幅
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # 计算RSI
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['factor']=df['RSI']

    return df

# 计算所有股票的因子
def calculate_all_factors(price_path,window=10):
    # 获取所有股票代码
    all_data_file = os.listdir(price_path)
    # 创建一个空的DataFrame来保存所有股票的因子
    all_factors = pd.DataFrame()

    # 遍历每一个股票代码，计算其因子，并添加到all_factors中
    for code in all_data_file:
        try:
            df=pd.read_csv(price_path+'\\'+code)
            #对每一个数据都计算第一步定义的算法
            factor = calculate_factor(df,window)
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

    #设置计算因子的数据路径
    price_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
    # 计算所有股票的因子,并且合并在一个表中
    all_factors = calculate_all_factors(price_path)
    all_factors=all_factors.dropna()

    # 对因子进行去除异常值和标准化操作
    all_factors = winsorize_and_standardize(all_factors)

    # 将结果保存到CSV文件中
    filename='rsi'+str(10)+'D'
    
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库"+"\\"+filename+'.csv',index=False)

# 如果这个脚本被作为主程序运行，则运行main()函数
if __name__ == "__main__":
    main()

