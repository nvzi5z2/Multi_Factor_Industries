import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


# 设置文件路径
path = r'D:\量化交易构建\ETF轮动策略\因子IC序列'

# 获取文件列表
file_list = os.listdir(path)

def calculate_IC_corr(path, file_list):
    # 读取因子文件并计算相关性
    results = []
    for i in file_list:
        data = pd.read_csv(os.path.join(path, i), index_col=[0])
        data.index = pd.to_datetime(data.index)
        data.columns = [i]
        results.append(data)
    
    results = pd.concat(results, axis=1)
    corr = results.corr()
    return corr

# 计算相关性矩阵
Corr = calculate_IC_corr(path, file_list)

# 绘制热力图
plt.figure(figsize=(10, 8))  # 设置图形的尺寸
sns.heatmap(Corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('IC Correlation Heatmap')  # 设置图形标题
plt.show()



def set_clusters(Corr_df,n_groups):
    # 假设 Corr_df 是已经计算好的相关性矩阵
    # 转换相关性为距离
    distance_matrix = 1 - Corr_df

    # 执行层次聚类
    linked = linkage(distance_matrix, 'average')

    # 绘制树状图来观察聚类情况
    dendrogram(linked, labels=Corr_df.index)
    plt.title('Dendrogram')
    plt.xlabel('Strategy Index')
    plt.ylabel('Distance')
    # plt.axhline(y=1.4, color='r', linestyle='--')  # 添加一条红线表示截断位置
    plt.show()

    # # 根据树状图选择的截断值进行聚类
    # clusters = fcluster(linked, 1.4, criterion='distance')

    # 直接指定生成三个聚类
    clusters = fcluster(linked, n_groups, criterion='maxclust')

    # 将聚类结果添加到原始 DataFrame 中
    Corr_df['Cluster'] = clusters

    # 输出聚类结果
    print(Corr_df['Cluster'])

    # 创建一个字典来存储每个聚类的成员列表
    cluster_dict = {}
    for cluster_id in np.unique(clusters):
        cluster_dict[cluster_id] = list(Corr_df.index[Corr_df['Cluster'] == cluster_id])

    # 输出每个聚类的成员
    for key, value in cluster_dict.items():
        print(f"Cluster {key}: {value}")

    return cluster_dict

# groups=set_clusters(Corr,9)