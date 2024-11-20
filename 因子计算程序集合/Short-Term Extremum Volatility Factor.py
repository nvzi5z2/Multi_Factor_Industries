# 导入需要的模块
import os
import pandas as pd
import numpy as np
from scipy.stats import mstats, zscore
from tqdm import tqdm


# 计算因子的函数
def calculate_factor(df,window_1,window_2):

    """
    计算 Alpha#1 因子值

    参数:
    - df: 包含股票数据的 DataFrame，必须包含 'time' 和 'close' 列

    返回:
    - 包含 Alpha#1 因子值的 DataFrame
    """
    # 确保数据按照时间排序
    df = df.sort_values('time')
    
    # 计算每日收益率
    df['returns'] = df['close'].pct_change()
    
    # 计算N日收益率的标准差
    df['stddev'] = df['returns'].rolling(window=window_1).std()
    
    # 计算条件表达式结果
    df['signed_power'] = np.where(df['returns'] < 0, df['stddev'], df['close']) ** 2
    
    # 计算 signed_power 的N日最大值的索引
    df['ts_argmax'] = df['signed_power'].rolling(window=window_2).apply(np.argmax, raw=True)
    
    # 将 ts_argmax_5 的值进行排名
    df['rank'] = df['ts_argmax'].rank(pct=True)
    
    # 计算因子值
    df['factor'] = df['rank'] - 0.5

    return df

# 计算所有股票的因子
def calculate_all_factors(price_path,window_1=10,window_2=5):
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

    #设置计算因子的数据路径
    price_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
    # 计算所有股票的因子,并且合并在一个表中
    all_factors = calculate_all_factors(price_path)
    all_factors=all_factors.dropna()

    # 对因子进行去除异常值和标准化操作
    all_factors = winsorize_and_standardize(all_factors)

    # 将结果保存到CSV文件中
    filename='short_term_volatility'+str(10)+'D_'+str(5)+'D'
    
    all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库"+"\\"+filename+'.csv',index=False)


# 如果这个脚本被作为主程序运行，则运行main()函数
if __name__ == "__main__":
    main()

