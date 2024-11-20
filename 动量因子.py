import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
def setting_up():

    #设置画图字体为黑体
    plt.rcParams['font.sans-serif'] = 'simhei'

    #正常显示负号在图里
    plt.rcParams['axes.unicode_minus']=False

    return
setting_up()


benchmark='000300.SH'

day=60

begin_date='2021-01-01'

end_date='2022-08-04'

class Factors_of_Momentum():

    def __init__(self,code,Day):

        self.code=code

        self.Day=Day

        self.signal=signal

        self.benchmark=benchmark

        self.begin_date=begin_date

        self.end_date=end_date

        self.Alpha_U=Alpha_U

        self.Alpha_L=Alpha_L

        self.df=df

    def LLT(Code,Day):

            def Get_Index_Close_Price(Code):

                filepath="C:\\Users\\Wesle\\Desktop\\量化交易构建\\个股选择模型\\大类资产配置模型\\data\\个股行情"+"\\"+Code+".csv"

                df=pd.read_csv(filepath,index_col=[0])

                df=df[["ths_close_price_stock"]]

                df.columns=[Code]

                return df

            Close_Price_Data=Get_Index_Close_Price(Code).dropna()

            Close_Price_Series=Close_Price_Data.iloc[:,0]

            #计算LLT指标
            llt=Close_Price_Series.copy()

            alpha=2/(Day+1)

            for i in range(2,len(Close_Price_Series)):

                llt[i]=(alpha-(alpha**2)/4)*Close_Price_Series[i]+((alpha**2)/2)*Close_Price_Series[i-1]-\
                       (alpha-(3*(alpha**2))/4)*Close_Price_Series[i-2]+2*(1-alpha)*llt[i-1]-((1-alpha)**2)*llt[i-2]

            Close_Price_Data.loc[:,"LLT"]=llt

            def Caculate_Slope(Close_Price_Data):

                LLT_Series=Close_Price_Data.loc[:,"LLT"]

                Slopes=[]

                for i in range(0,len(LLT_Series)):

                    Slope=(LLT_Series[i]/LLT_Series[i-1])-1

                    Slopes.append(Slope)
                return Slopes

            LLT_Slope=Caculate_Slope(Close_Price_Data)

            Close_Price_Data.loc[:,"LLT_Slope"]=LLT_Slope

            return Close_Price_Data

    def LLT_Strategy(Code, Day):
            def LLT(Code, Day):

                def Get_Index_Close_Price(Code):
                
                    filepath="C:\\Users\\Wesle\\Desktop\\量化交易构建\\个股选择模型\\大类资产配置模型\\data\\个股行情"+"\\"+Code+".csv"

                    df=pd.read_csv(filepath,index_col=[0])

                    df=df[["ths_close_price_stock"]]

                    df.columns=[Code]
                    
                    return df

                Close_Price_Data = Get_Index_Close_Price(Code).dropna()

                Close_Price_Series = Close_Price_Data.iloc[:, 0]

                # 计算LLT指标
                llt = Close_Price_Series.copy()

                alpha = 2 / (Day + 1)

                for i in range(2, len(Close_Price_Series)):
                    llt[i] = (alpha - (alpha ** 2) / 4) * Close_Price_Series[i] + ((alpha ** 2) / 2) * Close_Price_Series[
                        i - 1] - \
                             (alpha - (3 * (alpha ** 2)) / 4) * Close_Price_Series[i - 2] + 2 * (1 - alpha) * llt[i - 1] - (
                                         (1 - alpha) ** 2) * llt[i - 2]

                Close_Price_Data.loc[:, "LLT"] = llt

                def Caculate_Slope(Close_Price_Data):

                    LLT_Series = Close_Price_Data.loc[:, "LLT"]

                    Slopes = []

                    for i in range(0, len(LLT_Series)):
                        Slope =(LLT_Series[i] / LLT_Series[i - 1])-1

                        Slopes.append(Slope)

                    return Slopes

                LLT_Slope = Caculate_Slope(Close_Price_Data)

                Close_Price_Data.loc[:, "LLT_Slope"] = LLT_Slope

                return Close_Price_Data

            LLT_Data = LLT(Code, Day)

            LLT_Signal = LLT_Data.loc[:, "LLT_Slope"]

            for i in range(0, len(LLT_Signal)):

                if LLT_Signal[i] > 0:

                    LLT_Signal[i] = 100

                elif LLT_Signal[i] < 0:

                    LLT_Signal[i] = 50

                else:
                    LLT_Signal[i] = LLT_Signal[i - 1]

            LLT_Signal = pd.DataFrame(LLT_Signal.values, index=LLT_Data.index, columns=["LLT信号"])

            return LLT_Signal

    def LLT_Strategy_Alphalimit(Code, Day, Alpha_U, Alpha_L):

            def LLT(Code, Day):

                def Get_Index_Close_Price(Code):

                    filepath = "C:\\Users\\Wesle\\Desktop\\量化交易构建\\个股选择模型\\大类资产配置模型\\data\\指数行情" + "\\" + Code + ".xlsx"

                    df = pd.read_excel(filepath, index_col=[0])

                    return df

                Close_Price_Data = Get_Index_Close_Price(Code).dropna()

                Close_Price_Series = Close_Price_Data.iloc[:, 0]

                # 计算LLT指标
                llt = Close_Price_Series.copy()

                alpha = 2 / (Day + 1)

                for i in range(2, len(Close_Price_Series)):
                    llt[i] = (alpha - (alpha ** 2) / 4) * Close_Price_Series[i] + ((alpha ** 2) / 2) * \
                             Close_Price_Series[
                                 i - 1] - \
                             (alpha - (3 * (alpha ** 2)) / 4) * Close_Price_Series[i - 2] + 2 * (1 - alpha) * llt[
                                 i - 1] - (
                                     (1 - alpha) ** 2) * llt[i - 2]

                Close_Price_Data.loc[:, "LLT"] = llt

                def Caculate_Slope(Close_Price_Data):

                    LLT_Series = Close_Price_Data.loc[:, "LLT"]

                    Slopes = []

                    for i in range(0, len(LLT_Series)):
                        Slope = (LLT_Series[i] / LLT_Series[i - 1]) - 1

                        Slopes.append(Slope)

                    return Slopes

                LLT_Slope = Caculate_Slope(Close_Price_Data)

                Close_Price_Data.loc[:, "LLT_Slope"] = LLT_Slope

                return Close_Price_Data

            LLT_Data = LLT(Code, Day)

            LLT_Signal = LLT_Data.loc[:, "LLT_Slope"]

            for i in range(0, len(LLT_Signal)):

                if LLT_Signal[i] > Alpha_U:

                    LLT_Signal[i] = 100

                elif LLT_Signal[i] < Alpha_L:

                    LLT_Signal[i] = 0

                else:
                    LLT_Signal[i] = LLT_Signal[i - 1]

            LLT_Signal = pd.DataFrame(LLT_Signal.values, index=LLT_Data.index, columns=["LLT信号"])

            LLT_Signal = LLT_Signal

            return LLT_Signal

    def Backtest_for_Single_Factor(signal, benchmark, begin_date, end_date):
            # 获取benchmark价格数据

        def get_index_data(Code):

                filepath="C:\\Users\\Wesle\\Desktop\\量化交易构建\\个股选择模型\\大类资产配置模型\\data\\个股行情"+"\\"+Code+".csv"

                df=pd.read_csv(filepath,index_col=[0])

                df=df[["ths_close_price_stock"]]

                df.columns=[Code]
                    
                return df

        benchmark_close_price = get_index_data(benchmark)

        benchmark_pct = benchmark_close_price / benchmark_close_price.shift(1) - 1

        strategy_position = benchmark_pct.iloc[:, 0] * (signal.iloc[:, 0] / 100)

        strategy_position.index=pd.to_datetime(strategy_position.index)

        Strategy_NV = strategy_position + 1

        Strategy_NV = Strategy_NV[begin_date:end_date].dropna()

        Strategy_NV.iloc[0] = 1

        Strategy_NV = Strategy_NV.cumprod()

        benchmark_NV = benchmark_pct + 1

        benchmark_NV.index=pd.to_datetime(benchmark_NV.index)

        benchmark_NV = benchmark_NV.loc[begin_date:end_date, :]

        benchmark_NV.iloc[0, :] = 1

        benchmark_NV = benchmark_NV.cumprod()

        benchmark_NV.loc[:, "回测净值"] = Strategy_NV

        return benchmark_NV

    def Analyzing_Performance(df):
            # 总收益
            total_return = (df.iloc[-1, :] / df.iloc[0, :]) - 1

            total_return.index = ['总收益率']

            period_number = df.shape[0]

            # 年化收益率
            annual_return = ((total_return + 1) ** (250 / period_number)) - 1

            annual_return.index = ["年化收益率"]

            # 年化波动率

            pct_chg = df / df.shift(1) - 1

            pct_chg = pct_chg.dropna()

            dayily_std = pct_chg.std()

            annual_std = dayily_std * (250 ** 0.5)

            annual_std.index = ["年化波动率"]

            # 最大回撤

            def MaxDrawdown(df):
                i = np.argmax((np.maximum.accumulate(df) - df) / np.maximum.accumulate(df))

                j = np.argmax(df[:i])

                return (1 - df[i] / df[j])

            maxdrawdown = MaxDrawdown(df.iloc[:, 0])

            maxdrawdown = pd.Series(maxdrawdown, index=["最大回撤"])

            # sharp_ratio

            Sharp_ratio = annual_return[0] / annual_std[0]

            Sharp_ratio = pd.Series(Sharp_ratio, index=["夏普比率"])

            # karma_ratio

            Karma_ratio = annual_return[0] / maxdrawdown[0]

            Karma_ratio = pd.Series(Karma_ratio, index=['卡玛比率'])

            # 合并

            result = pd.concat([total_return, annual_return, annual_std, maxdrawdown, Sharp_ratio, Karma_ratio])

            result = pd.DataFrame(result.values, index=result.index, columns=[df.columns])

            return result

    def LLT_Test(Begin_Date,End_Date,LLT_Declay):

        Code_List=os.listdir("C:\\Users\\Wesle\\Desktop\\量化交易构建\\个股选择模型\\大类资产配置模型\\data\\个股行情")

        PFS=[]
        
        for i in Code_List:

            try:

                Code=i.replace(".csv","")

                Signal=FM.LLT_Strategy(Code,LLT_Declay)

                Signal=Signal.shift(1)

                Back_test=FM.Backtest_for_Single_Factor(Signal,Code,Begin_Date,End_Date)

                PF_BT=FM.Analyzing_Performance(Back_test[["回测净值"]])

                Holding_BT=FM.Analyzing_Performance(Back_test[[Back_test.columns[0]]])

                Total_PF=pd.merge(PF_BT,Holding_BT,right_index=True,left_index=True)

                Total_PF=Total_PF.loc["夏普比率":"卡玛比率",:]

                Total_PF.loc[:,"绩效差距："+Code]=Total_PF.loc[:,"回测净值"].values-Total_PF.loc[:,Code].values

                Total_PF=Total_PF[["绩效差距："+Code]]

                Total_PF=Total_PF.T

                PFS.append(Total_PF)

            except:

                print(i+"出现错误")
        
        PFS=pd.concat(PFS,axis=0)

        Sharp_Win=PFS[PFS["夏普比率"]>0]

        Karma_Win=PFS[PFS["卡玛比率"]>0]

        Sharp_mean=PFS[["夏普比率"]].mean()

        Karma__mean=PFS[["卡玛比率"]].mean()
        
        print('夏普胜率='+str(len(Sharp_Win)/len(PFS)))

        print("卡玛胜率="+str(len(Karma_Win)/len(PFS)))

        print("平均夏普提高"+str(Sharp_mean))

        print("平均卡玛提高"+str(Karma__mean))

        return PFS

