import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TimeSeriesGCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(TimeSeriesGCN, self).__init__()
        self.gcn1 = GCNConv(num_features, 16)
        self.gcn2 = GCNConv(16, 32)
        self.conv2d = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, edge_index):
        # GCN layers
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)

        # 转换为 [batch_size, channels, seq_len, length]
        x = x.unsqueeze(0)  # 添加一个批次维度
        x = x.transpose(1, 2)  # 交换维度，形状变为 [batch_size, seq_len, channels, 1]

        # 2D Convolution layer
        x = self.conv2d(x)  # 2D 卷积操作

        # 池化层，将特征图转换为一维特征向量
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # 假设时间序列数据的特征数为10，序列长度为5，类别数为3
    batch_size = 4
    seq_len = 5
    num_features = 10
    num_classes = 3
    model = TimeSeriesGCN(num_features, num_classes)

    # 假设我们有4个样本，每个样本有5个时间步，每个时间步有10个特征
    features = torch.randn(batch_size, seq_len, num_features)  # 批次大小为4的节点特征

    # 构建图的边缘索引
    # 这里只是一个示例，您需要根据您的图结构来定义边缘
    # 假设每个节点都与下一个节点相连
    edge_index = torch.tensor([[0, 1, 2, 3, 4],
                               [1, 2, 3, 4, 0]], dtype=torch.long).t().contiguous()

    # 为了处理批次数据，我们需要创建一个batch矢量
    batch = torch.arange(0, batch_size).unsqueeze(0).repeat(seq_len, 1)

    # 前向传播
    out = model(features, edge_index)
    print(out)