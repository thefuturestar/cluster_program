import numpy as np
import matplotlib.pyplot as plt

# ��ȡ OPTICS ���
def load_optics_result(filepath):
    data = np.loadtxt(filepath, delimiter=",")  

    # �������ǰ�������������ݣ��������Ǿ����ǩ���������ǿɴ����
    if data.shape[1] < 4:
        raise ValueError("Error: File format incorrect. Expecting at least 4 columns.")
    
    # ǰ��������������
    coordinates = data[:, :2]
    # �������� labels�������ţ�
    labels = data[:, 2].astype(int)
    # �������ǿɴ����
    reach_dists = data[:, 3]
    
    # ����������ȷ������ orders�����������ǰ�˳�����еģ�
    orders = np.arange(len(labels))  # ���� 0,1,2,3,... ������

    return orders, coordinates, labels, reach_dists

# ��ʾ����ͼ
def plotReachability(reach_dists, eps):
    plt.figure()
    plt.plot(range(len(reach_dists)), reach_dists, label="Reachability Distance")
    plt.axhline(y=eps, color='r', linestyle='--', label=f"Eps = {eps}")
    plt.xlabel("Data Points (ordered)")
    plt.ylabel("Reachability Distance")
    plt.title("Reachability Plot")
    plt.legend()
    plt.show()

# ��ʾ������
def plotFeature(data, labels):
    clusterNum = len(set(labels))
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)

    # ȷ�� labels ά��ƥ�� data
    if len(labels) != len(data):
        full_labels = np.full(len(data), -1)  # ����Ĭ�ϱ�ǩ
        full_labels[orders] = labels  # �޸� labels
    else:
        full_labels = labels

    print(f"data shape: {data.shape}, labels shape: {full_labels.shape}")

    for i in range(-1, clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[full_labels == i]  # ȷ��������ȷ
        ax.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=12)

    plt.show()

# ������
if __name__ == "__main__":
    # �޸Ĵ�·��Ϊ��Ľ���ļ�·��
    result_filepath = "F:/vivado/Vivado_Project/cluster_alpha/cluster_alpha.sim/sim_1/behav/xsim/cluster_results.csv"
    raw_data_filepath = "F:/vivado/Vivado_Project/cluster_alpha/cluster_alpha.sim/sim_1/cluster.csv"

    # ��ȡԭʼ����
    raw_data = np.loadtxt(raw_data_filepath, delimiter=",")

    # ��ȡ OPTICS ���
    orders, coordinates, labels, reach_dists = load_optics_result(result_filepath)

    # ���� Reachability Plot
    plotReachability(reach_dists, eps=3)

    # ����������
    plotFeature(raw_data, labels)
