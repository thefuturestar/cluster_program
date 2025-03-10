import numpy as np
import matplotlib.pyplot as plt

# 读取 OPTICS 结果
def load_optics_result(filepath):
    data = np.loadtxt(filepath, delimiter=",")  

    # 这里假设前两列是坐标数据，第三列是聚类标签，第四列是可达距离
    if data.shape[1] < 4:
        raise ValueError("Error: File format incorrect. Expecting at least 4 columns.")
    
    # 前两列是坐标数据
    coordinates = data[:, :2]
    # 第三列是 labels（聚类编号）
    labels = data[:, 2].astype(int)
    # 第四列是可达距离
    reach_dists = data[:, 3]
    
    # 重新生成正确的索引 orders（假设数据是按顺序排列的）
    orders = np.arange(len(labels))  # 生成 0,1,2,3,... 的索引

    return orders, coordinates, labels, reach_dists

# 显示决策图
def plotReachability(reach_dists, eps):
    plt.figure()
    plt.plot(range(len(reach_dists)), reach_dists, label="Reachability Distance")
    plt.axhline(y=eps, color='r', linestyle='--', label=f"Eps = {eps}")
    plt.xlabel("Data Points (ordered)")
    plt.ylabel("Reachability Distance")
    plt.title("Reachability Plot")
    plt.legend()
    plt.show()

# 显示分类结果
def plotFeature(data, labels):
    clusterNum = len(set(labels))
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)

    # 确保 labels 维度匹配 data
    if len(labels) != len(data):
        full_labels = np.full(len(data), -1)  # 创建默认标签
        full_labels[orders] = labels  # 修复 labels
    else:
        full_labels = labels

    print(f"data shape: {data.shape}, labels shape: {full_labels.shape}")

    for i in range(-1, clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[full_labels == i]  # 确保索引正确
        ax.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=12)

    plt.show()

# 主函数
if __name__ == "__main__":
    # 修改此路径为你的结果文件路径
    result_filepath = "F:/vivado/Vivado_Project/cluster_alpha/cluster_alpha.sim/sim_1/behav/xsim/cluster_results.csv"
    raw_data_filepath = "F:/vivado/Vivado_Project/cluster_alpha/cluster_alpha.sim/sim_1/cluster.csv"

    # 读取原始数据
    raw_data = np.loadtxt(raw_data_filepath, delimiter=",")

    # 读取 OPTICS 结果
    orders, coordinates, labels, reach_dists = load_optics_result(result_filepath)

    # 画出 Reachability Plot
    plotReachability(reach_dists, eps=3)

    # 画出聚类结果
    plotFeature(raw_data, labels)
