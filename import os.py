import os
import pandas as pd


import os
import pandas as pd

def calculate_f_score(codes, date, data_path):
    """
    根据输入的股票代码列表和日期，计算Piotroski F-score，并返回结果DataFrame。
    
    参数:
    - codes: 股票代码列表 (如 ['600570.SH', '000001.SZ'])
    - date: 选择的日期 (如 '2022/1/14')
    - data_path: 数据存储路径 (如 'D:/量化交易构建/市场数据库/数据库/个股财务数据库')

    返回:
    - DataFrame: 包含股票代码和计算出的F-score
    """
    f_score_list = []  # 用于存储每只股票的F-score
    
    # 将输入的日期转换为datetime格式
    date = pd.to_datetime(date)
    
    # 遍历每只股票代码
    for code in codes:
        file_path = os.path.join(data_path, f"{code}.csv")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在，跳过该股票。")
            continue
        
        # 读取股票数据
        df = pd.read_csv(file_path)
        
        # 将time列转换为datetime格式，并设置为index
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # 按日期先排序
        df = df.sort_index(ascending=True)
        
        # 找到当前日期的数据
        if date not in df.index:
            print(f"股票 {code} 在日期 {date} 没有数据，跳过该股票。")
            continue
        
        df_current = df.loc[date]
        
        # 获取当前日期的索引
        current_index = df.index.get_loc(date)
        
        # 检查过去第252条数据是否存在
        if current_index < 252:
            print(f"股票 {code} 在过去252个交易日内没有足够的数据，跳过该股票。")
            continue
        
        # 取得过去第252条数据（前一年左右的数据）
        df_previous = df.iloc[current_index - 252]
        
        # 获取当前和前一年的财务数据
        net_income = df_current['ths_np_atoopc_pit_stock']  # 归属于母公司所有者的净利润
        total_assets = df_current['ths_total_assets_pit_stock']  # 总资产
        cash_flow = df_current['ths_ncf_from_oa_pit_stock']  # 经营活动产生的现金流量净额
        total_liab = df_current['ths_total_liab_pit_stock']  # 总负债
        current_assets = df_current['ths_total_current_assets_pit_stock']  # 流动资产
        current_liab = df_current['ths_total_current_liab_pit_stock']  # 流动负债
        shares = df_current['ths_total_shares_stock']  # 总股本
        revenue = df_current['ths_operating_total_revenue_stock']  # 营业总收入
        gross_margin = df_current['ths_gross_selling_rate_stock']  # 销售毛利率
        
        net_income_prev = df_previous['ths_np_atoopc_pit_stock']
        total_assets_prev = df_previous['ths_total_assets_pit_stock']
        total_liab_prev = df_previous['ths_total_liab_pit_stock']
        current_assets_prev = df_previous['ths_total_current_assets_pit_stock']
        current_liab_prev = df_previous['ths_total_current_liab_pit_stock']
        revenue_prev = df_previous['ths_operating_total_revenue_stock']
        gross_margin_prev = df_previous['ths_gross_selling_rate_stock']

        # 初始化F-score为0
        f_score = 0
        
        # 盈利能力 (4项)
        ROA = net_income / total_assets  # 计算ROA
        ROA_prev = net_income_prev / total_assets_prev  # 前一年ROA
        if net_income > 0:
            f_score += 1  # 净利润为正
        if cash_flow > 0:
            f_score += 1  # 经营现金流为正
        if ROA > ROA_prev:
            f_score += 1  # 资产回报率较前一年提高
        if cash_flow > net_income:
            f_score += 1  # 经营现金流大于净利润
        
        # 杠杆、流动性和融资状况 (3项)
        if total_liab / total_assets < total_liab_prev / total_assets_prev:
            f_score += 1  # 杠杆比率较前一年下降
        current_ratio = current_assets / current_liab
        current_ratio_prev = current_assets_prev / current_liab_prev
        if current_ratio > current_ratio_prev:
            f_score += 1  # 流动比率较前一年提高
        if shares == df_previous['ths_total_shares_stock']:
            f_score += 1  # 没有发新股
        
        # 经营效率 (2项)
        if gross_margin > gross_margin_prev:
            f_score += 1  # 毛利率较前一年提高
        asset_turnover = revenue / total_assets
        asset_turnover_prev = revenue_prev / total_assets_prev
        if asset_turnover > asset_turnover_prev:
            f_score += 1  # 资产周转率较前一年提高
        
        # 将结果存储到列表
        f_score_list.append({
            'code': code,
            'F-score': f_score
        })
    
    # 将结果转换为DataFrame
    result_df = pd.DataFrame(f_score_list)
    
    return result_df

# 示例使用
codes = ['600570.SH', '000001.SZ']  # 股票代码列表
date = '2024/9/30'  # 选择的日期
data_path = 'D:/量化交易构建/市场数据库/数据库/个股财务数据库'  # 数据存储路径

f_score_df = calculate_f_score(codes, date, data_path)
print(f_score_df)