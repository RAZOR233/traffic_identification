from paddle.io import Dataset
import pandas as pd
import numpy as np

# 用于将pd dataframe转换成为矩阵
def construct_matrix(data, block_size=5, resolution=128):
    # 缩放包到达时间与包长范围大小一致
    data.arrival_time = data.arrival_time - data.arrival_time.min()
    time_scale = 3000
    packet_range = [-1500, 1500]
    data.arrival_time = data.arrival_time / (block_size * 1000) * time_scale
    # 构建统计矩阵
    hist_bins = resolution
    bin_range = [[0, time_scale], packet_range]
    hist = np.histogram2d(data.arrival_time, data.packet_size, bins=hist_bins, range=bin_range)
    flow_matrix = hist[0]
    assert flow_matrix.max() > 0.0, 'Zero Matrix!'
    # 归一化
    flow_matrix = flow_matrix / flow_matrix.max()
    flow_matrix = flow_matrix.astype(np.float32)
    return flow_matrix
# 实现flowpic数据集类
class FlowPicSet(Dataset):
    # 从标签csv中读取标签以及数据存放path
    def __init__(self, path):
        data = pd.read_csv(path)
        self.matrices = data['path']
        self.labels = data['label']

    #返回单条数据
    def __getitem__(self, index):
        """
        实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        # matrix表示一条流对应的矩阵
        matrix = construct_matrix(pd.read_csv(self.matrices[index]))
        matrix = matrix[np.newaxis,:,:]
        label = np.array([self.labels[index]])

        return matrix, label

    def __len__(self):
        return len(self.labels)