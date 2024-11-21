import pandas as pd
import alphalens as al
import matplotlib
from pathlib import Path
import tqdm
import os 
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from tqdm import tqdm


#读取因子数据并整合
def read_factor_data(data_path):
    # 读取因子数据
    data = pd.read_csv(data_path)
    data=data.dropna()
    data.loc[:,"time"]=pd.to_datetime(data.loc[:,"time"])
    factor_data=data[['time','thscode','standard_factor']]
    factor_data.loc[:,"time"]=pd.to_datetime(factor_data.loc[:,"time"])
    factor_data.set_index(['time', 'thscode'], inplace=True)
    factor_data.index=factor_data.index.set_names(['factor_date','secucode'])
    result = factor_data.sort_index()

    return result

#读取价格数据并整合
def read_price_data(file_path):
    #读取价格数据
    path=file_path
    list=os.listdir(path)
    all_prices=[]
    for i in list:
        df=pd.read_csv(path+'\\'+i,index_col=[0])
        name=i.replace('.csv',"")
        df.index=pd.to_datetime(df.index)
        close=df[['close']]
        close.columns=[name]
        all_prices.append(close)
    all_prices=pd.concat(all_prices,axis=1)
    all_prices.sort_index(inplace=True)
    all_prices=all_prices.dropna()

    #行情数据归一
    all_prices=all_prices.pct_change()
    all_prices=all_prices+1
    all_prices.iloc[0,:]=1
    result=all_prices.cumprod()
    result.index=pd.to_datetime(result.index)

    return result


#初步筛选因子的结果表
# def select_factors(file_list):

#     stats_list = []  # 初始化一个空列表以存储每个因子的统计数据
#     for i in tqdm(range(0,len(file_list))):
#         try:
#             #读取因子数据
#             datapath=r'D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库'
#             data_path=datapath+'\\'+file_list[i]
#             factor=read_factor_data(data_path)
#             factor_name = file_list[i].replace('.csv','')


#             #读取价格数据
#             file_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
#             price=read_price_data(file_path)

#             #合并为符合alphalens结构的数据


#             merged_data = al.utils.get_clean_factor_and_forward_returns(factor=factor, prices=price, periods=(1,5,10,20,30),quantiles=10,max_loss=1)


#         # 计算信息系数 (IC)
#             ic = al.performance.factor_information_coefficient(merged_data)

#             # 计算其他统计量
#             ic_mean = ic.mean()
#             ic_std = ic.std()
#             risk_adjusted_ic = ic_mean / ic_std
#             t_stats = ic.apply(lambda x: stats.ttest_1samp(x.dropna(), 0)[0])
#             p_values = ic.apply(lambda x: stats.ttest_1samp(x.dropna(), 0)[1])
#             ic_skew = ic.skew()
#             ic_kurtosis = ic.apply(lambda x: x.kurtosis())

#             # 创建 DataFrame
#             stats_df = pd.DataFrame({
#                 'Factor': factor_name,
#                 'IC Mean': ic_mean,
#                 'IC Std.': ic_std,
#                 'Risk-Adjusted IC': risk_adjusted_ic,
#                 't-stat(IC)': t_stats,
#                 'p-value(IC)': p_values,
#                 'IC Skew': ic_skew,
#                 'IC Kurtosis': ic_kurtosis
#             }).reset_index()
            
#             # 将结果添加到列表中
#             stats_list.append(stats_df)
#         except:

#             print(i+'出现错误跳过')

#         # 合并结果到一个 DataFrame
#         combined_stats_df = pd.concat(stats_list, ignore_index=True)
#         combined_stats_df=combined_stats_df.sort_values(by='Risk-Adjusted IC')


#     return combined_stats_df

# file_list=os.listdir(r'D:\量化交易构建\ETF轮动策略\行业轮动策略\待测试因子库')

# result=select_factors(file_list)

# # os.chdir(r'D:\量化交易构建\ETF轮动策略\ETF轮动策略程序')
# # result.to_excel('优化结果.xlsx')

#选择合适的因子进行仔细研究

def main():
    #读取因子数据
    datapath=r'D:\量化交易构建\ETF轮动策略\待测试因子库'
    factor_file='rank_vwap.csv'
    data_path=datapath+'\\'+factor_file
    factor=read_factor_data(data_path)
    #读取价格数据
    file_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
    price=read_price_data(file_path)

    #合并为符合alphalens结构的数据

    merged_data = al.utils.get_clean_factor_and_forward_returns(factor=factor, prices=price, periods=(10,),quantiles=5,max_loss=1)

    #生成测试结果
    al.tears.create_full_tear_sheet(merged_data)

    #生成因子表现图

    #1.找出因子每日收益率，按组从高到低排序的收益率

    quantile_returns, quantile_returns_std_err = al.performance.mean_return_by_quantile(
        merged_data,
        by_date=True,
        by_group=False)
    #2.绘画不同分组的等权的累计收益率曲线

    al.plotting.plot_cumulative_returns_by_quantile(quantile_returns['10D'], '1D', ax=None)


    #3.绘画最高组与最低组的超额曲线

    mean_returns_spread,std=al.performance.compute_mean_returns_spread(quantile_returns, upper_quant=5, lower_quant=1, std_err=None)

    al.plotting.plot_mean_quantile_returns_spread_time_series(mean_returns_spread['10D'], std_err=None, bandwidth=1, ax=None)
    

    #4.绘画按照因子值作为权重的全资产多空收益

    factor_weights_returns=al.performance.factor_returns(merged_data, demeaned=True, group_adjust=False, equal_weight=False, by_asset=False)

    al.plotting.plot_cumulative_returns(factor_weights_returns['10D'], period='1D', title=None, ax=None)

    #输出IC序列用来和别的因子做相关性测试

    factor_information_coefficient=al.performance.factor_information_coefficient(merged_data, group_adjust=False, by_group=False)

    for i in factor_information_coefficient.columns:

        ic_ts=factor_information_coefficient[[i]]

        file_name=factor_file.replace('.csv','')
        
        export_path=r'D:\量化交易构建\ETF轮动策略\因子IC序列'
        
        os.chdir(export_path)

        ic_ts.to_csv(file_name+'['+i+']'+'.csv')
        
        print('因子IC值输出完毕')


main()


def export_selected_factor_ic(datapath):

    factor_list=os.listdir(datapath)

    for i in factor_list:
        
        factor_file=i
        data_path=datapath+'\\'+factor_file
        factor=read_factor_data(data_path)
                #读取价格数据
        file_path=r'D:\量化交易构建\市场数据库\数据库\申万二级行业量价数据\1D'
        price=read_price_data(file_path)
        
    #合并为符合alphalens结构的数据

        merged_data = al.utils.get_clean_factor_and_forward_returns(factor=factor, prices=price, periods=(10,),quantiles=5,max_loss=1)

        #输出IC序列用来和别的因子做相关性测试

        factor_information_coefficient=al.performance.factor_information_coefficient(merged_data, group_adjust=False, by_group=False)

        for i in factor_information_coefficient.columns:

            ic_ts=factor_information_coefficient[[i]]

            file_name=factor_file.replace('.csv','')
            
            export_path=r'D:\量化交易构建\ETF轮动策略\因子IC序列'
            
            os.chdir(export_path)

            ic_ts.to_csv(file_name+'['+i+']'+'.csv')
            
            print('因子IC值输出完毕')

    return print('全部IC输出完毕')

datapath=r'D:\量化交易构建\ETF轮动策略\行业轮动因子库\合成因子选项'
# export_selected_factor_ic(datapath)