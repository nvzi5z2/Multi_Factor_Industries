import pandas as pd
import glob
import os
from scipy.stats import zscore

def winsorize_and_standardize(df):
    """
    对因子值进行极值处理和标准化。
    
    参数:
    - df: 包含因子值的DataFrame，必须包含列 'time' 和 'composite_standard_factor'
    
    返回:
    - 处理后的DataFrame
    """
    def limit_to_std(x):
        mean = x.mean()
        std = x.std()
        # 将超过3倍标准差的值设为3倍标准差
        x = x.apply(lambda v: min(max(v, mean - 3 * std), mean + 3 * std))
        return x

    # 应用limit_to_std函数处理每个时间组的因子值
    df['composite_standard_factor'] = df.groupby('time')['composite_standard_factor'].transform(limit_to_std)
    
    # 使用zscore方法进行标准化
    df['composite_standard_factor'] = df.groupby('time')['composite_standard_factor'].transform(zscore)
    
    return df

def combine_factors_from_directory(directory):
    """
    将指定目录下的所有因子文件的标准化因子值等权合成为一个复合因子文件，
    并确保对齐所有文件中的time和thscode列。
    
    参数:
    - directory: 包含因子CSV文件的目录路径
    
    返回:
    - 包含复合因子值的DataFrame
    """
    # 获取指定目录下所有CSV文件的路径
    file_paths = glob.glob(os.path.join(directory, "*.csv"))
    
    combined_df = None
    
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        if combined_df is None:
            combined_df = df[['time', 'thscode']].copy()
            combined_df['composite_standard_factor'] = df['standard_factor']
        else:
            # 合并当前文件的标准化因子值
            combined_df = combined_df.merge(
                df[['time', 'thscode', 'standard_factor']],
                on=['time', 'thscode'],
                how='inner'
            )
            # 累加标准化因子值
            combined_df['composite_standard_factor'] += combined_df['standard_factor']
            # 删除已合并的列
            combined_df.drop(columns=['standard_factor'], inplace=True)
    
    # 计算标准化因子值的平均值
    combined_df['composite_standard_factor'] /= len(file_paths)
    
    # 对合成的因子进行极值处理和标准化
    # combined_df = winsorize_and_standardize(combined_df)
    
    # 将列名统一为 'standard_factor'
    combined_df.rename(columns={'composite_standard_factor': 'standard_factor'}, inplace=True)
    
    return combined_df

# 指定包含因子文件的目录路径
directory = "D:\量化交易构建\ETF轮动策略\行业轮动策略\ETF因子库\合成因子选项"
composite_df = combine_factors_from_directory(directory)


#输出文件路径
# # 将合成的因子DataFrame保存到新的CSV文件中
export_path=r'D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库'
output_path = os.path.join(export_path, "result_8.csv")
composite_df.to_csv(output_path, index=False)