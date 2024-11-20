import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from numpy import nan as NA
import matplotlib.pyplot as plt
import math
from pandas.tseries.offsets import Day,MonthEnd,BDay
import os
from scipy import stats
from scipy.stats import percentileofscore
from scipy.stats import rankdata, mstats, zscore
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm


def setting_up():

    #设置画图字体为黑体
    plt.rcParams['font.sans-serif'] = 'simhei'

    #正常显示负号在图里
    plt.rcParams['axes.unicode_minus']=False

    return
setting_up()


class ETF_Strategy():

    def __init__(self,End_Date,Code_List,Begin_Date,Data_Path_List):

        self.Code_List=Code_List

        self.End_Date=End_Date

        self.Begin_Date=Begin_Date

        self.Data_Path_List=Data_Path_List

    def second_order_momentum_factor(window_1=20,window_2=10,window_3=5):
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
        def calculate_all_factors(price_path,window_1,window_2,window_3):
            # 获取所有股票代码
            all_data_file = os.listdir(price_path)
            # 创建一个空的DataFrame来保存所有股票的因子
            all_factors = pd.DataFrame()

            # 遍历每一个股票代码，计算其因子，并添加到all_factors中
            for code in all_data_file:
                try:
                    df=pd.read_csv(price_path+'\\'+code)
                    #对每一个数据都计算第一步定义的算法
                    factor = calculate_factor(df,window_1,window_2,window_3)
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

        #设置计算因子的数据路径
        price_path=r'D:\量化交易构建\市场数据库\数据库\同花顺ETF跟踪指数量价数据'
        # 计算所有股票的因子,并且合并在一个表中
        all_factors = calculate_all_factors(price_path,window_1,window_2,window_3)
        all_factors=all_factors.dropna()

        # 对因子进行去除异常值和标准化操作
        all_factors = winsorize_and_standardize(all_factors)

        # 将结果保存到CSV文件中
        filename='second_order_momentum_factor'+str(window_1)+'D_'+str(window_2)+'D_'+str(window_3)+'D'

        all_factors.to_csv(r"D:\量化交易构建\ETF轮动策略\ETF轮动策略程序\ETF因子库\有效因子"+"\\"+filename+'.csv')

        return all_factors

    def Second_Order_Momentum_Factor(Code, Window1, Window2, Window):
        # 设置数据路径
        path = "C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库\\同花顺ETF跟踪指数量价数据\\"
        file = os.path.join(path, "{}.csv".format(Code))

        # 读取数据  
        data = pd.read_csv(file)

        # 确保数据是按日期排序的
        data['time'] = pd.to_datetime(data['time'])
        data = data.sort_values('time')
        data.set_index('time', inplace=True)

        # 计算过去Window1日的均价
        data['MA'] = data['close'].rolling(window=Window1).mean()

        # 计算最新一期收盘价与过去一段时间均价的偏离度
        data['Deviation'] = (data['close'] - data['MA']) / data['MA']

        # 计算（最新一期收盘价与过去一段时间均价的偏离度）-过去window2日的（最新一期收盘价与过去一段时间均价的偏离度） 
        data['Deviation_Diff'] = data['Deviation'] - data['Deviation'].shift(Window2)

        # 将上述的结果进行指数加权移动平均，平均日数量为Window日
        data['Second_Order_Momentum'] = data['Deviation_Diff'].ewm(span=Window).mean()

        # 将结果列重命名为 '<Code>二阶动量分数'
        data.rename(columns={'Second_Order_Momentum': '{}二阶动量分数'.format(Code)}, inplace=True)

        # 返回只包含二阶动量分数的DataFrame
        return data[['{}二阶动量分数'.format(Code)]]
    
    #动量因子
    def Momentum_Term_Spread_Factor(code, window1, window2):
        # 读取数据
        filepath = f"C:/Users/Wesle/Desktop/量化交易构建/市场数据库/数据库/同花顺ETF跟踪指数量价数据/{code}.csv"
        data = pd.read_csv(filepath)
        
        # 确保数据是按时间升序排列的
        data = data.sort_values(by='time')

        # 计算长期动量和短期动量
        data['long_term_momentum'] = (data['close'] - data['close'].shift(window1)) / data['close'].shift(window1)
        data['short_term_momentum'] = (data['close'] - data['close'].shift(window2)) / data['close'].shift(window2)

        # 计算动量期限差因子
        data[code + '动量期限差分数'] = data['long_term_momentum'] - data['short_term_momentum']

        # 返回结果，只包含日期和动量期限差因子
        return data.set_index('time')[[code + '动量期限差分数']]

    def Caculate_Momentum_Term_Spread_Score(End_Date,window1=150,window2=75):
        # 定义一个函数计算单个股票的动量期限差因子
        #经过测试75 25的参数最好，且这个因子有效
        def Momentum_Term_Spread_Factor(code, window1, window2):
            # 指定股票数据文件的路径
            filepath = f"C:/Users/Wesle/Desktop/量化交易构建/市场数据库/数据库/同花顺ETF跟踪指数量价数据/{code}.csv"
            # 读取股票数据
            data = pd.read_csv(filepath)
            # 按时间排序
            data = data.sort_values(by='time')

            # 计算长期动量
            data['long_term_momentum'] = (data['close'] - data['close'].shift(window1)) / data['close'].shift(window1)
            # 计算短期动量
            data['short_term_momentum'] = (data['close'] - data['close'].shift(window2)) / data['close'].shift(window2)
            # 计算动量期限差因子
            data['Momentum_Term_Spread'] = data['long_term_momentum'] - data['short_term_momentum']

            # 返回带有动量期限差因子的数据
            return data.set_index('time')[['Momentum_Term_Spread']]

        # 指定长期和短期动量的窗口期
        window1 = window1
        window2 = window2

        # 指定要提取因子值的日期
        date = End_Date

        # 列出目录中的所有csv文件
        files = os.listdir('C:/Users/Wesle/Desktop/量化交易构建/市场数据库/数据库/同花顺ETF跟踪指数量价数据')

        # 初始化一个空的dataframe来存储所有股票的因子值
        factor_values = pd.DataFrame()

        for file in files:

            try:
                # 获取股票代码
                code = file.rstrip('.csv')
                # 计算股票的动量期限差因子
                factor = Momentum_Term_Spread_Factor(code, window1, window2)

                factor.index=pd.to_datetime(factor.index)
                # 提取指定日期的因子值并添加到dataframe中
                factor_value = factor.loc[date, 'Momentum_Term_Spread']

                factor_values.loc[code, 'Momentum_Term_Spread'] = factor_value

            except:

                print('出现错误跳过')

        # 对因子值进行Winsorize处理以移除异常值
        factor_values['Momentum_Term_Spread'] = mstats.winsorize(factor_values['Momentum_Term_Spread'], limits=[0.01, 0.01])

        # 使用z-score对因子值进行标准化
        factor_values['Momentum_Term_Spread'] = zscore(factor_values['Momentum_Term_Spread'])

        return factor_values

    def Caculate_Second_Order_Momentum_Factor(End_Date):

        #定义一个函数计算单个股票的二阶动量因子值,从计算结果看这是一个反转因子

        def Second_Order_Momentum_Factor(Code, Window1, Window2, Window):
            # 设置数据路径
            path = "C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库\\同花顺ETF跟踪指数量价数据\\"
            file = os.path.join(path, "{}.csv".format(Code))

            # 读取数据  
            data = pd.read_csv(file)

            # 确保数据是按日期排序的
            data['time'] = pd.to_datetime(data['time'])
            data = data.sort_values('time')
            data.set_index('time', inplace=True)

            # 计算过去Window1日的均价
            data['MA'] = data['close'].rolling(window=Window1).mean()

            # 计算最新一期收盘价与过去一段时间均价的偏离度
            data['Deviation'] = (data['close'] - data['MA']) / data['MA']

            # 计算（最新一期收盘价与过去一段时间均价的偏离度）-过去window2日的（最新一期收盘价与过去一段时间均价的偏离度） 
            data['Deviation_Diff'] = data['Deviation'] - data['Deviation'].shift(Window2)

            # 将上述的结果进行指数加权移动平均，平均日数量为Window日
            data['Second_Order_Momentum'] = -(data['Deviation_Diff'].ewm(span=Window).mean())

            # 将结果列重命名为 '<Code>二阶动量分数'
            data.rename(columns={'Second_Order_Momentum': '{}二阶动量分数'.format(Code)}, inplace=True)

            # 返回只包含二阶动量分数的DataFrame
            return data[['{}二阶动量分数'.format(Code)]]

        date=End_Date

        Window1=25

        Window2=10

        Window=5

        # 列出目录中的所有csv文件
        files = os.listdir('C:/Users/Wesle/Desktop/量化交易构建/市场数据库/数据库/同花顺ETF跟踪指数量价数据')

        # 初始化一个空的dataframe来存储所有股票的因子值
        factor_values = pd.DataFrame()

        for file in files:

            try:
                # 获取股票代码
                code = file.rstrip('.csv')
                # 计算股票的二阶动量因子值
                factor = Second_Order_Momentum_Factor(code, Window1, Window2, Window)
                # 提取指定日期的因子值并添加到dataframe中
                factor_value = factor.loc[date, '{}二阶动量分数'.format(code)]
                factor_values.loc[code, 'Second_Order_Momentum_Factor'] = factor_value
            except:

                print('出现错误跳过')

        # 对因子值进行Winsorize处理以移除异常值
        factor_values['Second_Order_Momentum_Factor'] = mstats.winsorize(factor_values['Second_Order_Momentum_Factor'], limits=[0.01, 0.01])

        # 使用z-score对因子值进行标准化
        factor_values['Second_Order_Momentum_Factor'] = zscore(factor_values['Second_Order_Momentum_Factor'])

        # 打印所有股票的因子值
        return factor_values
   
    def Caculate_Information_Ratio_Factor(End_Date):

        def Caculate_Information_Ratio_Windway(Begin_Date, End_Date, Benchmark_Code='000906.SH'):

            def Get_ETF_Close_Price(code):

                filepath = "C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库\\同花顺ETF跟踪指数量价数据"

                data = pd.read_csv(filepath + "\\" + code+".csv", index_col=[0])

                data.index = pd.to_datetime(data.index)

                Close_Price = data[["close"]]

                Close_Price.columns=[code]

                return Close_Price

            Code_List_Path="C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库"

            Name="同花顺ETF跟踪指数代码库.xlsx"

            Code_List=pd.read_excel(Code_List_Path+"\\"+Name)

            Code_List=Code_List.loc[:,"跟踪指数代码"].tolist()

            Benchmark = Get_ETF_Close_Price(Benchmark_Code)

            IRs = []

            for i in range(0, len(Code_List)):

                try:

                    ETF_Close_Price = Get_ETF_Close_Price(Code_List[i])

                    Combined_Data = pd.merge(ETF_Close_Price, Benchmark, right_index=True, left_index=True)

                    Combined_Data = Combined_Data.loc[Begin_Date:End_Date, :]

                    # 计算区间内的超额收益率与超额波动率

                    Combined_Data_Total_Return = Combined_Data.iloc[-1, :] / Combined_Data.iloc[0, :] - 1

                    Days = len(Combined_Data.index)

                    # 计算区间内的年化收益率

                    Annual_Return = ((1 + Combined_Data_Total_Return) ** (250 / Days)) - 1

                    Alpha_Annual_Return = Annual_Return[0] - Annual_Return[1]

                    # 计算跟踪误差

                    Combined_Data_Pct = Combined_Data / Combined_Data.shift(1) - 1

                    Daily_Error = Combined_Data_Pct.loc[:, Code_List[i]] - Combined_Data_Pct.loc[:, Benchmark_Code]

                    Error = Daily_Error.std()

                    N = 250 ** 0.5

                    Annual_Error=Error*N

                    IR = Alpha_Annual_Return / Annual_Error

                    IR = pd.DataFrame(IR, index=[Code_List[i]], columns=["Information_Ratio"])

                    IRs.append(IR)

                    print("Information_Ratio计算完成度：" + str(i / len(Code_List)))

                except:

                    print(Code_List[i] + " 报错")

            IRs = pd.concat(IRs, axis=0)

            return IRs

        def Get_Crowdy_Ratio(End_Date):

            Code_List_Path="C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库"

            Name="同花顺ETF跟踪指数代码库.xlsx"

            Code_List=pd.read_excel(Code_List_Path+"\\"+Name)

            Code_List=Code_List.loc[:,"跟踪指数代码"].tolist()

            # 计算区间移动平均数的区间分位值

            def Get_Last_Percentile_Score(code,End_Date, Rolling_Mean=25):

                def Get_Index_Free_Turn_Data(code):
                    filepath = "C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库\\同花顺指数自由流通换手率"

                    data = pd.read_csv(filepath + "\\" + code+".csv", index_col=[0])

                    data.index = pd.to_datetime(data.index)

                    Free_Turn_Data = data[['ths_free_turnover_ratio_index']]

                    Free_Turn_Data.columns=[code]

                    return Free_Turn_Data

                Data = Get_Index_Free_Turn_Data(code)

                Data = Data.rolling(Rolling_Mean).mean()

                Final_Date=datetime.strptime(End_Date,'%Y-%m-%d')

                Previous_Date=Final_Date-timedelta(days=365)

                Begin_Date=Previous_Date.strftime('%Y-%m-%d')

                Score = stats.percentileofscore(Data.loc[Begin_Date:End_Date, Data.columns[0]],
                                                Data.loc[End_Date, Data.columns[0]])

                return Score

            Scores = []

            for i in range(0, len(Code_List)):

                try:
                    Score = Get_Last_Percentile_Score(Code_List[i],End_Date)

                    Score = pd.DataFrame(Score, index=[Code_List[i]],
                                            columns=["移动平均换手率区间百分位: " + End_Date])

                    print("换手率百分位计算完成率：" + str(i / len(Code_List)))

                    Scores.append(Score)
                
                except:

                    print(Code_List[i]+"出现错误")


            Scores = pd.concat(Scores, axis=0)

            return Scores

        Final_Date=datetime.strptime(End_Date,'%Y-%m-%d')

        Previous_Date=Final_Date-timedelta(days=90)

        Begin_Date=Previous_Date.strftime('%Y-%m-%d')
        
        IF=Caculate_Information_Ratio_Windway(Begin_Date,End_Date)

        IF['Information_Ratio']=mstats.winsorize(IF['Information_Ratio'], limits=[0.01, 0.01])

        IF['Information_Ratio'] = zscore(IF['Information_Ratio'])

        Crowdy=Get_Crowdy_Ratio(End_Date)

        IF.loc[:,"换手率拥挤度"]=Crowdy

        return IF

    # 均线偏离因子
    def MA_Deviation_Factor(code, window):
        # 构建文件路径
        file_path = os.path.join(
            'C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库\\同花顺ETF跟踪指数量价数据', code + '.csv')

        # 读取CSV文件
        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到。")
            return None

        # 确保数据按日期升序排列（如果文件中不是这样的话）
        data['time'] = pd.to_datetime(data['time'])
        data.sort_values('time', inplace=True)

        # 计算N日均线
        data[f'{window}day_MA'] = data['close'].rolling(window=window).mean()

        # 计算偏离因子
        data['Factor'] = (data['close'] - data[f'{window}day_MA']) / data[f'{window}day_MA']

        # 生成输出的DataFrame
        result_df = pd.DataFrame(data={'time': data['time'], 'Factor': data['Factor']})
        result_df.set_index('time', inplace=True)

        return result_df

    def Calculate_MA_Deviation_Factor(end_date, window=250):
        folder_path = 'C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库\\同花顺ETF跟踪指数量价数据'
        files = os.listdir(folder_path)
        results = []

        # 计算N日均线偏离因子并收集数据
        for file in files:

            try:
                if file.endswith('.csv'):
                    code = file.replace('.csv', '')
                    file_path = os.path.join(folder_path, file)
                    try:
                        data = pd.read_csv(file_path)
                    except FileNotFoundError:
                        print(f"文件 {file_path} 未找到。")
                        continue
                    data['time'] = pd.to_datetime(data['time'])
                    data.sort_values('time', inplace=True)
                    data[f'{window}day_MA'] = data['close'].rolling(window=window).mean()
                    data['Factor'] = (data['close'] - data[f'{window}day_MA']) / data[f'{window}day_MA']
                    factor_on_date = data.loc[data['time'] == pd.to_datetime(end_date), 'Factor']
                    results.append({'Code': code, 'MA_Deviation_Factor': factor_on_date.iloc[
                        0] if not factor_on_date.empty else np.nan})
            except:

                print(file + "出现问题")

        # 创建DataFrame
        factor_df = pd.DataFrame(results)
        factor_df.set_index('Code', inplace=True)

        # 去除异常值和标准化处理
        factor_df['MA_Deviation_Factor'] = mstats.winsorize(factor_df['MA_Deviation_Factor'], limits=[0.01, 0.01])
        factor_df['MA_Deviation_Factor'] = (factor_df['MA_Deviation_Factor'] - factor_df[
            'MA_Deviation_Factor'].mean()) / factor_df['MA_Deviation_Factor'].std()

        return factor_df

    #交易波动因子
    def Calculate_Amount_Volume_Std_Factor(end_date, window_size=500):

        def amount_volume_std_factor(df, window):
            df = df.sort_values('time')
            df['amount_std'] = df['amount'].rolling(window).std()
            df['volume_std'] = df['volume'].rolling(window).std()
            df = df.dropna()
            df['amount_factor'] = df[['amount_std']].transform(zscore)
            df['volume_factor'] = df[['volume_std']].transform(zscore)
            df['factor'] = -(df['amount_factor'] + df['volume_factor']) / 2
            return df

        # 处理极端值的函数
        def limit_to_std(x):
            mean = x.mean()
            std = x.std()
            x = x.apply(lambda v: min(max(v, mean - 3 * std), mean + 3 * std))
            return x

        # 文件夹路径
        folder_path = r'C:\Users\Wesle\Desktop\量化交易构建\市场数据库\数据库\同花顺ETF跟踪指数量价数据'
        files = os.listdir(folder_path)

        # 用于存放所有股票指定日因子值的列表
        factors_list = []

        # 遍历文件夹中的所有文件
        for file in files:
            try:
                file_path = os.path.join(folder_path, file)
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 计算因子值
                df_with_factors = amount_volume_std_factor(df, window_size)
                
                df_with_factors.index = pd.to_datetime(df_with_factors['time'])
                # 获取指定日期的因子值
                specified_date_factor = df_with_factors.loc[end_date:end_date]
                specified_date_factor = specified_date_factor[['thscode', 'factor']]
                factors_list.append(specified_date_factor)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        # 将因子值列表转换为DataFrame
        factors_df = pd.concat(factors_list)

        # 处理极端值
        factors_df['Amount_Volume_Std_Factor'] = limit_to_std(factors_df['factor'])

        # 标准化处理
        factors_df['Amount_Volume_Std_Factor'] = zscore(factors_df['Amount_Volume_Std_Factor'])

        # 将股票代码设置为索引
        factors_df.set_index('thscode', inplace=True)

        factors_df = factors_df[['Amount_Volume_Std_Factor']]

        return factors_df

    #换手率变化因子
    def Caculate_Turnover_Rate_Change_Factor(End_Date,Window1=125,Window2=75):

        def Turnover_Rate_Change_Factor(Code, Window1, Window2):
            # 根据股票代码设置文件路径
            file_path = f'C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库\\同花顺指数自由流通换手率\\{Code}.csv'

            # 读取csv文件
            data = pd.read_csv(file_path)
            data['time'] = pd.to_datetime(data['time'])
            data.set_index('time', inplace=True)

            # 计算滚动平均换手率
            data['rolling_mean_window1'] = data['ths_free_turnover_ratio_index'].rolling(window=Window1).mean()
            data['rolling_mean_window2'] = data['ths_free_turnover_ratio_index'].rolling(window=Window2).mean()

            # 计算因子值
            data['factor'] = data['rolling_mean_window1'] / data['rolling_mean_window2']

            return data[['factor']]

        # 设置文件夹路径
        folder_path = 'C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库\\同花顺指数自由流通换手率'

        # 获取所有的文件名
        files = os.listdir(folder_path)

        # 创建一个空的DataFrame来存储所有股票的因子值
        df_all = pd.DataFrame()

        # 对每一个文件计算因子值
        for file in files:
            try:
                # 获取股票代码
                code = file.replace('.csv',"")
                # 计算因子值
                df = Turnover_Rate_Change_Factor(code, Window1, Window2)
                # 提取指定日期的因子值
                factor_value = df.loc[End_Date, 'factor']
                # 添加到df_all
                df_all.loc[code, 'Turnover_Rate_Change_Factor'] = factor_value
            
            except:
                print('出现错误跳过')

        df_all=df_all.dropna()

        # 去除异常值
        df_all['Turnover_Rate_Change_Factor'] = mstats.winsorize(df_all['Turnover_Rate_Change_Factor'], limits=[0.01, 0.01])

        # z-score标准化
        df_all['Turnover_Rate_Change_Factor'] = zscore(df_all['Turnover_Rate_Change_Factor'])

        return df_all

    #多空对比因子

    def Calculate_Long_Short_Compare_Change_Factor(date,Window1=30, Window2=15):
            # 定义函数计算因子值 #
            #因子定义因子的定义为<首先计算多空力量对比。分子我们用多头力量与空头力量相减，即
        # (𝐶𝑙𝑜𝑠𝑒 − 𝐿𝑜𝑤) − (𝐻𝑖𝑔ℎ − 𝐶𝑙𝑜𝑠𝑒)，分母为最高价减去最低价，也就是日内价格区
        # 间的极值。再将所得多空力量对比乘上当日行业成交量，可得当日多空力量对比
        # 的金额绝对值。
        # 我们用长期每日多空力量对比的指数加权平均值，减去短期每日多空力量对比的
        # 指数加权平均值，可以得到近期多空力量对比相对于长期多空力量对比均值的变化。
        # 因子值越大，说明近期多头相对于空头力量减弱；因子值越小，说明近期多空对
        # 比度相对于长期加大
            def Long_Short_Compare_Change(Code):
                db_path = r"C:\Users\Wesle\Desktop\量化交易构建\市场数据库\数据库\同花顺ETF跟踪指数量价数据"
                data = pd.read_csv(os.path.join(db_path, f"{Code}.csv"))
                data['time'] = pd.to_datetime(data['time'])
                data.set_index('time', inplace=True)
                data['LS_Compare'] = data['volume'] * ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
                long_term = data['LS_Compare'].ewm(span=Window1).mean()
                short_term = data['LS_Compare'].ewm(span=Window2).mean()
                data['LS_Compare_Change'] = long_term - short_term
                # 添加处理逻辑以提取特定日期的因子值
                return data.loc[date, 'LS_Compare_Change'] if date in data.index else np.nan

            # 定义函数遍历所有股票文件并计算因子值
            def calc_factor_values():
                db_path = r"C:\Users\Wesle\Desktop\量化交易构建\市场数据库\数据库\同花顺ETF跟踪指数量价数据"
                codes = [f[:-4] for f in os.listdir(db_path) if f.endswith('.csv')]
                values = [Long_Short_Compare_Change(code) for code in codes]
                df = pd.DataFrame(values, index=codes, columns=['Long_Short_Compare_Change'])
                return df

            # 定义函数进行因子值的预处理
            def preprocess_factor_values(df):
                transformer = QuantileTransformer(output_distribution='normal')
                df['Long_Short_Compare_Change'] = transformer.fit_transform(df[['Long_Short_Compare_Change']])
                return df
            # 计算和处理因子值
            df = calc_factor_values()
            df=df.dropna()
            df = preprocess_factor_values(df)

            return df


    # ROE变化因子-30天调仓
    def Calculate_Roe_Change_Factor(End_Date,window=120):
        # 定义盈利预测数据的文件夹路径
        directory = "C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库\\同花顺ETF跟踪指数一致预期数据\\盈利预测综合值"
        
        # 获取文件夹中所有的CSV文件
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        
        # 初始化一个空的DataFrame用于存储结果
        result_df = pd.DataFrame(columns=['Code', 'roe_change_factor'])
        
        # 遍历所有的CSV文件
        for file in files:
            # 从文件名中获取股票代码（去掉.csv后缀）
            code = file[:-4]  
            
            # 构建每只股票的CSV文件路径
            filename = directory + "\\" + file
            
            # 读取CSV文件
            df = pd.read_csv(filename)
            
            # 按时间排序
            df = df.sort_values(by='time')
            
            # 将时间设置为索引
            df.set_index('time', inplace=True)
            
            # 计算预期 ROE 变化因子
            df['roe_change_factor'] = (df['ths_fore_roe_mean_index'] / df['ths_fore_roe_mean_index'].shift(window)) - 1
            
            # 如果指定日期的数据不存在，跳过此次循环
            if End_Date not in df.index:
                continue
            
            # 提取指定日期的因子值
            factor = df.loc[End_Date, 'roe_change_factor']
            
            # 如果因子值是NaN，跳过此次循环
            if pd.isnull(factor):
                continue
            
            # 将结果添加到结果DataFrame中
            result_df = result_df.append({'Code': code, 'roe_change_factor': factor}, ignore_index=True)
        
        qt = QuantileTransformer(output_distribution='normal')

        # 对数据进行转换
        transformed_factors = qt.fit_transform(result_df['roe_change_factor'].values.reshape(-1, 1))

        # 创建一个新的DataFrame来存储结果
        result = pd.DataFrame(transformed_factors, index=result_df['Code'], columns=['roe_change_factor'])

        # 更改列名为'Roe_Change_Factor'
        result.columns = ['Roe_Change_Factor']
        
        return result

        
        #超跌850均线因子
    #ERS变化因子-20天调仓
    def Calculate_Eps_Change_Factor(date,window=140) -> pd.DataFrame:

        # 数据的保存地址
        data_dir = "C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库\\同花顺ETF跟踪指数一致预期数据\\盈利预测综合值\\"

        # 初始化一个空的 DataFrame 来存储所有股票的结果
        results = pd.DataFrame()

        # 遍历每个文件
        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                # 计算每只股票的因子值
                data = pd.read_csv(os.path.join(data_dir, file), parse_dates=['time'])
                data.set_index('time', inplace=True)
                data['EPS Change'] = data['ths_fore_eps_index'].pct_change(periods=window)

                # 提取指定日期的因子值
                factor_value = data.loc[date, 'EPS Change'] if date in data.index else None

                # 添加到结果中
                results.loc[file[:-4], '预期 EPS百分比变化因子'] = factor_value

        # Winsorize 异常值
        results['预期 EPS百分比变化因子'] = mstats.winsorize(results['预期 EPS百分比变化因子'], limits=[0.05, 0.05])

        results.dropna(inplace=True)

        # z-score标准化处理
        results['预期 EPS百分比变化因子'] = zscore(results['预期 EPS百分比变化因子'])

        results.rename(columns={'预期 EPS百分比变化因子': 'EPS_Change_Factor'}, inplace=True)

        return results

    def Calculate_ROE_Two_MA_Deviation_Factor(End_Date,window1=120,window2=90):


    #波动率占比因子
     def Downside_Volatility_Ratio_MA_Factor(Code, window1, window2):
        # 从CSV文件中读取数据
        file_path = os.path.join(
            "C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库\\同花顺ETF跟踪指数量价数据",
            f"{Code}.csv")
        data = pd.read_csv(file_path)

        # 确保数据按时间排序
        data['time'] = pd.to_datetime(data['time'])
        data.sort_values('time', inplace=True)

        # 计算日收益率
        data['returns'] = data['close'].pct_change()

        # 计算总波动率
        data['total_volatility'] = data['returns'].rolling(window=window1).std()

        # 计算下行波动率
        downside_returns = data['returns'].apply(lambda x: x if x < 0 else 0)
        data['downside_volatility'] = downside_returns.rolling(window=window1).std()

        # 计算下行波动率占比因子
        data['downside_vol_ratio'] = data['downside_volatility'] / data['total_volatility']

        # 计算移动平均值
        data['downside_vol_ratio_MA'] = data['downside_vol_ratio'].rolling(window=window2).mean()
        #计算因子值，占比越小分数越高
        data['downside_vol_ratio_MA'] = 1 - data['downside_vol_ratio_MA']
        # 仅保留需要的列
        result = data[['time', 'downside_vol_ratio_MA']].set_index('time')

        return result

    def Caculate_Downside_Volatility_Ratio_MA_Factor(End_Date, window1=120, window2=25):
        def Downside_Volatility_Ratio_MA_Factor(Code, window1, window2):
            # 从CSV文件中读取数据
            file_path = os.path.join(
                "C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库\\同花顺ETF跟踪指数量价数据",
                f"{Code}.csv")
            data = pd.read_csv(file_path)

            # 确保数据按时间排序
            data['time'] = pd.to_datetime(data['time'])
            data.sort_values('time', inplace=True)

            # 计算日收益率
            data['returns'] = data['close'].pct_change()

            # 计算总波动率
            data['total_volatility'] = data['returns'].rolling(window=window1).std()

            # 计算下行波动率
            downside_returns = data['returns'].apply(lambda x: x if x < 0 else 0)
            data['downside_volatility'] = downside_returns.rolling(window=window1).std()

            # 计算下行波动率占比因子
            data['downside_vol_ratio'] = data['downside_volatility'] / data['total_volatility']

            # 计算移动平均值
            data['downside_vol_ratio_MA'] = data['downside_vol_ratio'].rolling(window=window2).mean()
            # 计算因子值，占比越小分数越高
            data['downside_vol_ratio_MA'] = 1 - data['downside_vol_ratio_MA']
            # 仅保留需要的列
            result = data[['time', 'downside_vol_ratio_MA']].set_index('time')

            return result



            return pd.Series(mstats.winsorize(series, limits=limits), index=series.index)

        def remove_extreme_values(df, column, num_std=3):
            mean_val = df[column].mean()
            std_val = df[column].std()
            lower_bound = mean_val - num_std * std_val
            upper_bound = mean_val + num_std * std_val
            df[column] = df[column].clip(lower_bound, upper_bound)
            return df

        directory = "C:\\Users\\Wesle\\Desktop\\量化交易构建\\市场数据库\\数据库\\同花顺ETF跟踪指数量价数据"
        factors = {}
        for filename in os.listdir(directory):
            try:
                if filename.endswith(".csv"):
                    code = filename[:-4]
                    df = Downside_Volatility_Ratio_MA_Factor(code, window1, window2)
                    factors[code] = df.loc[End_Date, 'downside_vol_ratio_MA']
            except Exception as e:
                print(f'出现错误跳过: {e}')

        df_factors = pd.DataFrame.from_dict(factors, orient='index', columns=['Downside_Volatility_Ratio_MA_Factor'])

        # 使用3σ原则去除极值
        df_factors = remove_extreme_values(df_factors, 'Downside_Volatility_Ratio_MA_Factor')

        # 使用Z-score标准化处理
        df_factors['Downside_Volatility_Ratio_MA_Factor'] = zscore(df_factors['Downside_Volatility_Ratio_MA_Factor'])
            
        return df_factors




def Mul_Factor_Strategy(End_Date):

    ETF=ETF_Strategy
    #动量因子
    MTSS=ETF.Caculate_Momentum_Term_Spread_Score(End_Date)
    #（10天）
    SOMF=ETF.Caculate_Second_Order_Momentum_Factor('2024-03-29')

    MA=ETF.Calculate_MA_Deviation_Factor(End_Date)
    
    Momentum=pd.concat([MTSS,MA],axis=1)

    Momentum.loc[:,'Total']=Momentum.loc[:,"Momentum_Term_Spread"]+ Momentum.loc[:,"MA_Deviation_Factor"] 
    
    Momentum=Momentum.dropna()

    Momentum=Momentum[['Total']]

    Momentum.columns=['Momentum']

    Momentum=zscore(Momentum)    

    #交易波动因子

    AV_STD_Factor=ETF.Calculate_Amount_Volume_Std_Factor(End_Date)

    AV_STD_Factor.columns=['Trade_Std']

    #换手率变化

    Turnover=ETF.Caculate_Turnover_Rate_Change_Factor(End_Date)
    Turnover=Turnover.dropna()

    #多空对比（10天调仓)

    LSCC=ETF.Calculate_Long_Short_Compare_Change_Factor('2024-03-29')
    LS_Total=LSCC

    #一致预期变化因子

    Expected=ETF.Caculate_Expected_Data_MoM(End_Date)
    Expected.columns=['Expected_MOM']
    Expected=Expected.dropna()

    ROE_Change=ETF.Calculate_Roe_Change_Factor(End_Date)
    ROE_Change.columns=['ROE_Change']
    ROE_Change=ROE_Change.dropna()

    EPS_Change=ETF.Calculate_Eps_Change_Factor(End_Date)
    EPS_Change.columns=['EPS_Change']
    EPS_Change=EPS_Change.dropna()

    Expected_Change_Total=pd.concat([ROE_Change,EPS_Change,Expected],axis=1)
    Expected_Change_Total.loc[:,"Expected_Change_Total"]=(Expected_Change_Total.loc[:,"ROE_Change"]\
                                                          +Expected_Change_Total.loc[:,"EPS_Change"]\
                                                            +Expected_Change_Total.loc[:,"Expected_MOM"])
    Expected_Change_Total=Expected_Change_Total[['Expected_Change_Total']]
    Expected_Change_Total=Expected_Change_Total.dropna()
    Expected_Change_Total=zscore(Expected_Change_Total)

    #下行波动率占比因子
    Downside_Volatility_Ratio_MA_Factor=ETF.Caculate_Downside_Volatility_Ratio_MA_Factor(End_Date)

    #汇总

    Total_Factor=pd.concat([Momentum,AV_STD_Factor,Turnover,LS_Total,Expected_Change_Total,Downside_Volatility_Ratio_MA_Factor,SOMF],axis=1)

    Total_Factor.loc[:,"Sum"]=Total_Factor.loc[:,"Momentum"]\
    +Total_Factor.loc[:,"Trade_Std"]+Total_Factor.loc[:,"Turnover_Rate_Change_Factor"]\
    +Total_Factor.loc[:,"Long_Short_Compare_Change"]+Total_Factor.loc[:,"Expected_Change_Total"]\
    +Total_Factor.loc[:,"Downside_Volatility_Ratio_MA_Factor"]+Total_Factor.loc[:,"Second_Order_Momentum_Factor"]

    Total_Factor=Total_Factor.sort_values(by='Sum',ascending=False)

    os.chdir(r'C:\Users\Wesle\Desktop\量化交易构建\ETF轮动策略\ETF轮动策略程序\result')
    Total_Factor.to_excel('量价多因子选股.xlsx')

    return Total_Factor



class select_invest_target():

    def Select_ETF_and_Build_Portfilo(self):

        Dirc=pd.read_excel(r'C:\Users\Wesle\Desktop\量化交易构建\ETF轮动策略\ETF轮动策略程序\result'+"\\量价多因子选股.xlsx",index_col=[0])

        #选出前30

        Top_30=Dirc.iloc[:31,:]

        #选出对应的ETF代码和指数

        Top_30_Code=Top_30.index.to_list()

        #对应ETF名称文档

        ETF_Name=pd.read_excel(r'C:\Users\Wesle\Desktop\量化交易构建\ETF轮动策略\ETF轮动策略程序\result'+'\\ETF对应名称.xlsx')

        ETF_Name.index=ETF_Name.loc[:,"跟踪指数代码"]

        ETF_Codes=[]

        ETF_Names=[]

        for i in range(0,len(Top_30_Code)):
            
            try:

                Index_Code=Top_30_Code[i]

                ETF_Match_Code=ETF_Name.loc[[Index_Code]]

                if len(ETF_Match_Code.loc[:,"误差"])>1:

                    ETF_Match_Code=ETF_Match_Code.sort_values(by='误差',ascending=True)

                    ETF_Code=ETF_Match_Code.iloc[0,0]

                    Name=ETF_Match_Code.iloc[0,1]

                else:
                    
                    ETF_Code=ETF_Match_Code.iloc[0,0]

                    Name=ETF_Match_Code.iloc[0,1]

                ETF_Codes.append(ETF_Code)

                ETF_Names.append(Name)
            
            except:

                print(Top_30_Code[i])
        
        Result=pd.DataFrame(ETF_Codes,index=Top_30_Code,columns=["ETF对应指数"])

        Result.loc[:,"ETF_Name"]=ETF_Names

        Final_Result=pd.merge(Result,Dirc,right_index=True,left_index=True)

        os.chdir(r'C:\Users\Wesle\Desktop\量化交易构建\ETF轮动策略\ETF轮动策略程序\result')

        Final_Result.to_excel('量价ETF组合结果前30.xlsx')

        return Final_Result

    def Caculate_Port_Corr(self,index_code_list,window):
        
        path=r'C:\Users\Wesle\Desktop\量化交易构建\市场数据库\数据库\同花顺ETF跟踪指数量价数据'

        port=[]
        for i in index_code_list:

            data=pd.read_csv(path+'\\'+i+'.csv',index_col=[0])
            data.index=pd.to_datetime(data.index)
            close=data[['close']]
            close.columns=[i]
            port.append(close)
        
        port=pd.concat(port,axis=1)
        port=port.dropna()
        port=port.iloc[-window:,:]
        port_pct=port.pct_change()
        corr=port_pct.corr()

        return corr


ETF=ETF_Strategy

smf=ETF.second_order_momentum_factor()

# sit=select_invest_target()
# etf_result=sit.Select_ETF_and_Build_Portfilo()

# code_list=['931008.CSI','931166.CSI','931402.CSI','931187.CSI','931052.CSI','399998.SZ']

# Corr=sit.Caculate_Port_Corr(code_list,250)

