import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj_matrix):
        x = torch.matmul(adj_matrix, x)
        x = self.linear(x)
        return F.relu(x)  # F.sigmoid, F.tanh, F.silu, etc. can be used here


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gc2 = GraphConvolutionLayer(hidden_dim, output_dim)

    def forward(self, x, adj_matrix):
        x = self.gc1(x, adj_matrix)
        x = self.gc2(x, adj_matrix)
        return x


# 模拟数据
adj_matrix = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
features = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
labels = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)

# 初始化模型
model = GCN(input_dim=2, hidden_dim=16, output_dim=1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(100):
    optimizer.zero_grad()
    output = model(features, adj_matrix)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item()}')
