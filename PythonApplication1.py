'''
import numpy as np
import matplotlib.pyplot as plt
import time
import operator
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
def compute_squared_EDM(X):
  return squareform(pdist(X,metric='euclidean'))    # 返回观测点对之间的距离矩阵
# 显示决策图
def plotReachability(data,eps):
    plt.figure()
    plt.plot(range(0,len(data)), data)
    plt.plot([0, len(data)], [eps, eps])
    plt.show()

# 显示分类的类别
def plotFeature(data,labels):
    clusterNum = len(set(labels))
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(-1, clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[np.where(labels == i)]
        ax.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=12)
    plt.show()
def updateSeeds(seeds,core_PointId,neighbours,core_dists,reach_dists,disMat,isProcess):
    # 获得核心点core_PointId的核心距离
    core_dist=core_dists[core_PointId]
    # 遍历core_PointId 的每一个邻居点
    for neighbour in neighbours:
        # 如果neighbour没有被处理过，计算该核心距离
        if(isProcess[neighbour]==-1):
            # 首先计算改点的针对core_PointId的可达距离
            new_reach_dist = max(core_dist, disMat[core_PointId][neighbour])
            if(np.isnan(reach_dists[neighbour])):   # 检测非数字元素
                reach_dists[neighbour]=new_reach_dist
                seeds[neighbour] = new_reach_dist
            elif(new_reach_dist<reach_dists[neighbour]):
                reach_dists[neighbour] = new_reach_dist
                seeds[neighbour] = new_reach_dist
    return seeds
def OPTICS(data,eps=np.inf,minPts=15):
    # 获得距离矩阵
    orders = []
    disMat = compute_squared_EDM(data)
    # 获得数据的行和列(一共有n条数据)
    n, m = data.shape
    # np.argsort(disMat)[:,minPts-1] 按照距离进行 行排序 找第minPts个元素的索引
    # disMat[np.arange(0,n),np.argsort(disMat)[:,minPts-1]] 计算minPts个元素的索引的距离
    temp_core_distances = disMat[np.arange(0,n),np.argsort(disMat)[:,minPts-1]]
    # 计算核心距离
    core_dists = np.where(temp_core_distances <= eps, temp_core_distances, -1)
    # 将每一个点的可达距离未定义
    reach_dists= np.full((n,), np.nan)
    # 将矩阵的中小于minPts的数赋予1，大于minPts的数赋予零，然后1代表对每一行求和,然后求核心点坐标的索引
    core_points_index = np.where(np.sum(np.where(disMat <= eps, 1, 0), axis=1) >= minPts)[0]
    # 用于标识是否被处理，没有被处理，设置为-1
    isProcess = np.full((n,), -1)
    # 遍历所有的核心点
    for pointId in core_points_index:
        # 如果核心点未被分类，将其作为的种子点，开始寻找相应簇集
        if (isProcess[pointId] == -1):
            # 将点pointId标记为当前类别(即标识为已操作)
            isProcess[pointId] = 1
            orders.append(pointId)
            # 寻找种子点的eps邻域且没有被分类的点，将其放入种子集合
            neighbours = np.where((disMat[:, pointId] <= eps) & (disMat[:, pointId] > 0) & (isProcess == -1))[0]
            seeds = dict()
            seeds=updateSeeds(seeds,pointId,neighbours,core_dists,reach_dists,disMat,isProcess)
            while len(seeds)>0:
                nextId = sorted(seeds.items(), key=operator.itemgetter(1))[0][0]
                del seeds[nextId]
                isProcess[nextId] = 1
                orders.append(nextId)
                # 寻找newPoint种子点eps邻域（包含自己）
                # 这里没有加约束isProcess == -1，是因为如果加了，本是核心点的，可能就变成了非核心点
                queryResults = np.where(disMat[:, nextId] <= eps)[0]
                if len(queryResults) >= minPts:
                    seeds=updateSeeds(seeds,nextId,queryResults,core_dists,reach_dists,disMat,isProcess)
                # 簇集生长完毕，寻找到一个类别
    # 返回数据集中的可达列表，及其可达距离
    return orders,reach_dists
def extract_dbscan(data,orders, reach_dists, eps):
    # 获得原始数据的行和列
    n,m=data.shape
    # reach_dists[orders] 将每个点的可达距离，按照有序列表排序（即输出顺序）
    # np.where(reach_dists[orders] <= eps)[0]，找到有序列表中小于eps的点的索引，即对应有序列表的索引
    reach_distIds=np.where(reach_dists[orders] <= eps)[0]
    # 正常来说：current的值的值应该比pre的值多一个索引。如果大于一个索引就说明不是一个类别
    pre=reach_distIds[0]-1
    clusterId=0
    labels=np.full((n,),-1)
    for current in reach_distIds:
        # 正常来说：current的值的值应该比pre的值多一个索引。如果大于一个索引就说明不是一个类别
        if(current-pre!=1):
            # 类别+1
            clusterId=clusterId+1
        labels[orders[current]]=clusterId
        pre=current
    return labels
data = np.loadtxt("data/cluster.csv", delimiter=",")
start = time.perf_counter()
orders,reach_dists=OPTICS(data,np.inf,30)
end = time.perf_counter()
print('finish all in %s' % str(end - start))
labels=extract_dbscan(data,orders,reach_dists,3)
plotReachability(reach_dists[orders],3)
plotFeature(data,labels)
'''


import numpy as np
import matplotlib.pyplot as plt
import time
import operator
from scipy.spatial.distance import pdist, squareform
import pandas as pd  # 新增用于保存 CSV 文件

# 计算欧几里得距离矩阵
def compute_squared_EDM(X):
    return squareform(pdist(X, metric='euclidean'))  # 返回点对距离矩阵

# 画出可达距离图（Reachability Plot）
def plotReachability(data, eps):
    plt.figure()
    plt.plot(range(len(data)), data, label="Reachability Distance")
    plt.axhline(y=eps, color='r', linestyle='--', label=f"Eps = {eps}")
    plt.xlabel("Data Points (ordered)")
    plt.ylabel("Reachability Distance")
    plt.title("Reachability Plot")
    plt.legend()
    plt.show()

# 画出聚类结果
def plotFeature(data, labels):
    clusterNum = len(set(labels))
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(-1, clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[labels == i]
        ax.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=12)
    plt.show()

# 更新种子点
def updateSeeds(seeds, core_PointId, neighbours, core_dists, reach_dists, disMat, isProcess):
    core_dist = core_dists[core_PointId]  # 核心点的核心距离
    for neighbour in neighbours:
        if isProcess[neighbour] == -1:  # 仅处理未访问过的点
            new_reach_dist = max(core_dist, disMat[core_PointId][neighbour])    # 计算新的可达距离
            if np.isnan(reach_dists[neighbour]):  # 如果尚未定义
                reach_dists[neighbour] = new_reach_dist
                seeds[neighbour] = new_reach_dist
                #print(f"新点: {neighbour}, 可达距离: {new_reach_dist}")
            elif new_reach_dist < reach_dists[neighbour]:  # 更新更小的可达距离
                reach_dists[neighbour] = new_reach_dist
                seeds[neighbour] = new_reach_dist
                #print(f"更新点: {neighbour}, 可达距离: {new_reach_dist}")
    #print(f"seeds_count: {len(seeds)}")
    
    return seeds

# OPTICS 算法
def OPTICS(data, eps=np.inf, minPts=15):
    orders = []
    disMat = compute_squared_EDM(data)  # 计算距离矩阵
    n, _ = data.shape   # 数据的行和列数

    # 计算核心距离
    sort_start = time.perf_counter()
    sorted_indices = np.argsort(disMat)  # 按行排序
    sort_end = time.perf_counter()
    global sort_time
    sort_time += (sort_end - sort_start)
    temp_core_distances = disMat[new_func(n), sorted_indices[:, minPts - 1]]  # 每个点的第 minPts 近距离
    core_dists = np.where(temp_core_distances <= eps, temp_core_distances, -1)  # 核心距离，如果小于等于 eps，取距离，否则为 -1

    reach_dists = np.full(n, np.nan)  # 初始化可达距离
    core_points_index = np.where(np.sum(np.where(disMat <= eps, 1, 0), axis=1) >= minPts)[0]    # 核心点索引
    isProcess = np.full(n, -1)  # -1 表示未处理

    # 遍历所有核心对象
    print(f"核心点索引数量: {len(core_points_index)}")
    for pointId in core_points_index:
        #print(f"当前处理核心点: {pointId}")
        if isProcess[pointId] == -1:
            isProcess[pointId] = 1
            orders.append(pointId)  # 将核心对象添加到结果中

            neighbours = np.where((disMat[:, pointId] <= eps) & (disMat[:, pointId] > 0) & (isProcess == -1))[0]
            seeds = dict()
            seeds = updateSeeds(seeds, pointId, neighbours, core_dists, reach_dists, disMat, isProcess)

            #output_file0 = "cluster_results0.csv"
            #df = pd.DataFrame(data, columns=["X", "Y"])
            #df["Reachability_Distance"] = reach_dists  # 添加可达距离列
            #df["Core_Distance"] = core_dists  # 添加核心距离列
            #df["Core_points_index"] = core_points_index  # 添加核心点索引列
            #df.to_csv(output_file0, index=False)

            while_count = 0
            update_count = 0
            while len(seeds) > 0:
                #print(f"当前种子点数量: {len(seeds)}")
                #print(f"seed(0)= {list(seeds.items())[0]}")
                sorted_seeds = timed_sort(seeds.items(), key=operator.itemgetter(1))
                nextId = sorted_seeds[0][0]
                #nextId = sorted(seeds.items(), key=operator.itemgetter(1))[0][0]    # 按可达距离对seeds中的点排序并选择排序后第一个项的索引
                del seeds[nextId]
                isProcess[nextId] = 1
                #print(f"当前处理点: {nextId},已标记isProcess")
                orders.append(nextId)
                while_count += 1
                #print(f"第 {while_count} 次循环")

                queryResults = np.where(disMat[:, nextId] <= eps)[0]    # 找到与nextId距离小于等于eps的所有点的索引
                if len(queryResults) >= minPts:     # 如果满足条件，nextId是一个核心点
                    seeds = updateSeeds(seeds, nextId, queryResults, core_dists, reach_dists, disMat, isProcess)
                    update_count += 1
                    #print(f"更新次数: {update_count},新核心点：{nextId}")

    return orders, reach_dists

def new_func(n):
    return np.arange(0, n)

# 提取 DBSCAN 样式的聚类结果
def extract_dbscan(data, orders, reach_dists, eps):
    n, _ = data.shape   # 获取数据的行数
    reach_distIds = np.where(reach_dists[orders] <= eps)[0] # 找到可达距离小于 eps 的点的索引
    pre = reach_distIds[0] - 1  # 初始化前一个索引
    clusterId = 0   # 初始化类别 ID
    labels = np.full(n, -1) # 初始化标签数组，未分类表示为-1

    for current in reach_distIds:
        if current - pre != 1:  # 如果索引不连续，说明是新的类别
            clusterId += 1
        labels[orders[current]] = clusterId
        pre = current

    return labels

#排序计时函数
def timed_sort(items, key):
    """对排序操作计时的函数"""
    global sort_time  # 使用全局变量记录总排序时间
    start_time = time.perf_counter()
    result = sorted(items, key=key)
    end_time = time.perf_counter()
    sort_time += (end_time - start_time)
    return result

### ============ 运行 OPTICS 并保存 CSV 结果 ============ ###
# 读取数据
data = np.loadtxt("data/cluster.csv", delimiter=",")

# 初始化排序时间
sort_time = 0

# 运行 OPTICS 算法
start = time.perf_counter()
orders, reach_dists = OPTICS(data, eps=np.inf, minPts=30)
end = time.perf_counter()
total_time = end - start

print('OPTICS 运行时间: %s 秒' % str(total_time))
print('排序操作总时间: %s 秒' % str(sort_time))
print('排序时间占比: %.2f%%' % (sort_time/total_time * 100))

# 进行 DBSCAN 聚类
eps_dbscan = 3
labels = extract_dbscan(data, orders, reach_dists, eps_dbscan)

# 可视化结果
plotReachability(reach_dists[orders], eps_dbscan)
plotFeature(data, labels)

# **保存结果到 CSV**
output_file = "cluster_results.csv"
df = pd.DataFrame(data, columns=["X", "Y"])  # 创建 DataFrame
#df["X"] = np.around(df["X"], decimals=2)
#df["Y"] = np.around(df["Y"], decimals=3)
#df["Cluster"] = labels  # 添加聚类结果列
df["Reachability_Distance"] = reach_dists  # 添加可达距离列
df.to_csv(output_file, index=False)

print(f"聚类结果已保存至 {output_file}")
# **保存结果到 CSV**

