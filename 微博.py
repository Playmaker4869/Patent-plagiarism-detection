import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GATNet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, num_heads):
        super(GATNet, self).__init__()

        # 定义两层GAT卷积层，每层包含指定数量的注意力头
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels)

    def forward(self, x, edge_index):
        # 第一层GAT计算
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))

        # 将不同注意力头的结果进行拼接或求平均（这里假设是拼接）
        x = torch.cat([x[:, i::self.conv1.heads] for i in range(self.conv1.heads)], dim=1)

        # 第二层GAT计算
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


x = torch.randn(8, 20)
edge_index = torch.tensor([[0, 2, 4, 5],
                           [1, 3, 5, 7]], dtype=torch.int)
label = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
epochs = 10

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATNet(num_features=20, hidden_channels=8, out_channels=2, num_heads=8).to(device)

# 假设dataset是一个PyG数据集对象，包含了节点特征、边索引等信息
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环中的前向传播与反向传播部分
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    out = model(x.to(device), edge_index.to(device))
    # loss = F.nll_loss(out[dataset.train_mask], dataset.y[dataset.train_mask].to(device))
    # loss.backward()
    # optimizer.step()
