# 导入需要的模块
import os
import pandas as pd
import numpy as np
from scipy.stats import mstats, zscore,rankdata
from tqdm import tqdm

# 计算因子的函数
def calculate_factor(df,window_1):

    #读取自由流通换手率数据
    def read_free_turn_rate(code):
        path=r'D:\量化交易构建\市场数据库\数据库\同花顺指数自由流通换手率'

        data=pd.read_csv(path+'\\'+code+'.csv')

        return data
    #设置格式为日期格式
    df=df.sort_values(by='time')
    df=df.set_index('time',drop=True)
    #计算滚动波动率
    df['pct']=df['close'].pct_change()
    df['std']=df['pct'].rolling(window_1).std()
    code=df.iloc[1,0]
    free_turn_rate=read_free_turn_rate(code)
    free_turn_rate=free_turn_rate.set_index('time',drop=True)
    free_turn_rate=free_turn_rate[['ths_free_turnover_ratio_index']]
    combined_df=pd.merge(df,free_turn_rate,right_index=True,left_index=True)
    combined_df['mean_free_turn']=combined_df['ths_free_turnover_ratio_index'].rolling(window_1).mean()

    combined_df['factor'] = combined_df['mean_free_turn'] / combined_df['std']

    return combined_df

# 计算所有股票的因子
def calculate_all_factors(price_path,window_1):
    # 获取所有股票代码
    all_data_file = os.listdir(price_path)
    # 创建一个空的DataFrame来保存所有股票的因子
    all_factors = pd.DataFrame()

    # 遍历每一个股票代码，计算其因子，并添加到all_factors中
    for code in all_data_file:
        try:
            df=pd.read_csv(price_path+'\\'+code)
            #对每一个数据都计算第一步定义的算法
            factor = calculate_factor(df,window_1)
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

#清除文档
def clear_folder(folder_path):
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在")
        return

    # 遍历文件夹中的所有文件和文件夹
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # 如果是文件，删除文件
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"文件 '{file_path}' 已删除")
            # 如果是文件夹，递归删除文件夹中的内容
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"文件夹 '{file_path}' 已删除")
        except Exception as e:
            print(f"删除 '{file_path}' 时出错: {e}")
#主程序
def main():
    clear_folder(r'D:\量化交易构建\ETF轮动策略\ETF轮动策略程序\待测试因子库')
    # 设置计算因子的窗口大小
    for window_1 in tqdm(range(10,201,10)):
        try:
            # 设置计算因子的数据路径
            price_path=r'D:\量化交易构建\市场数据库\数据库\ETF回测数据库\量价'
            # 计算所有股票的因子，并且合并在一个表中
            all_factors = calculate_all_factors(price_path, window_1)
            all_factors = all_factors.dropna()
            all_factors.loc[:,"time"]=all_factors.index
            all_factors=all_factors.reset_index(drop=True)

            # 对因子进行去除异常值和标准化操作
            all_factors = winsorize_and_standardize(all_factors)

            # 将结果保存到CSV文件中
            filename='bbs'+str(window_1)+'D'
            
            all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\ETF轮动策略程序\待测试因子库"+"\\"+filename+'.csv', index=False)
        except:

            print('出现错误')

# 如果这个脚本被作为主程序运行，则运行main()函数
if __name__ == "__main__":
    main()