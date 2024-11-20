# 导入需要的模块
import os
import pandas as pd
import numpy as np
from scipy.stats import mstats, zscore
from tqdm import tqdm


# 计算因子的函数
def calculate_factor(df,window_1,window_2,window_3):

    # 确保数据按照时间排序
    df = df.sort_values('time')
    # 计算过去Window1日的均价
    df['MA'] = df['close'].rolling(window=window_1).mean()

    # 计算最新一期收盘价与过去一段时间均价的偏离度
    df['Deviation'] = (df['close'] - df['MA']) / df['MA']

    # 计算（最新一期收盘价与过去一段时间均价的偏离度）-过去window2日的（最新一期收盘价与过去一段时间均价的偏离度） 
    df['Deviation_Diff'] = df['Deviation'] - df['Deviation'].shift(window_2)

    # 将上述的结果进行指数加权移动平均，平均日数量为Window日
    df['factor'] = df['Deviation_Diff'].ewm(span=window_3).mean()

    return df

# 计算所有股票的因子
def calculate_all_factors(price_path,window_1=25,window_2=10,window_3=5):
    # 获取所有股票代码
    all_data_file = os.listdir(price_path)
    # 创建一个空的DataFrame来保存所有股票的因子
    all_factors = pd.DataFrame()

    # 遍历每一个股票代码，计算其因子，并添加到all_factors中
    for code in all_data_file:
        try:
            df=pd.read_csv(price_path+'\\'+code)
            #对每一个数据都计算第一步定义的算法
            factor = calculate_factor(df,window_1=window_1,window_2=window_2,window_3=window_3)
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
    # 设置计算因子的窗口大小

    #设置计算因子的数据路径
    price_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
    # 计算所有股票的因子,并且合并在一个表中
    all_factors = calculate_all_factors(price_path)
    all_factors=all_factors.dropna()

    # 对因子进行去除异常值和标准化操作
    all_factors = winsorize_and_standardize(all_factors)

    # 将结果保存到CSV文件中
    filename='second_order_momentum_factor'+str(25)+'D_'+str(10)+'D_'+str(5)+'D'
    
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库"+"\\"+filename+'.csv',index=False)

# 如果这个脚本被作为主程序运行，则运行main()函数
if __name__ == "__main__":
    main()

