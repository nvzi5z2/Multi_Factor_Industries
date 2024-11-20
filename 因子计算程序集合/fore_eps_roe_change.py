# 导入需要的模块
import os
import pandas as pd
import numpy as np
from scipy.stats import mstats, zscore,rankdata

# 计算因子的函数
def calculate_factor(df,window):
    
    df=df.sort_values(by='time')

    df['eps_change']=df['ths_fore_eps_index'].pct_change(window)
    df['roe_change']=df['ths_fore_roe_mean_index'].pct_change(window)
    #标准化因子值
    df=df.dropna()
    df['eps_factor']=zscore(df['eps_change'])
    df['roe_factor']=zscore(df['roe_change'])
    df['factor']=df['eps_factor']+df['roe_factor']

    return df

# 计算所有股票的因子
def calculate_all_factors(price_path,window):
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
            print(code+"已完成")
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

    price_path=r'D:\量化交易构建\市场数据库\数据库\ETF回测数据库\盈利预测'
    # 设置计算因子的窗口大小
    for window in range(10,201,10):
        try:
        #设置计算因子的数据路径

            all_factors = calculate_all_factors(price_path,window)

            all_factors=all_factors.dropna()

        # 对因子进行去除异常值和标准化操作
            all_factors = winsorize_and_standardize(all_factors)

        # 将结果保存到CSV文件中
            filename='fore_eps_roe_change'+str(window)+'D'
        
            all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\ETF轮动策略程序\待测试因子库"+"\\"+filename+'.csv', index=False)
        except:

            print('出现错误')

# 如果这个脚本被作为主程序运行，则运行main()函数
if __name__ == "__main__":
    main()