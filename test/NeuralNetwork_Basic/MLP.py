import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 避免多个 OpenMP 运行时库实例的问题

import sys
project_path = os.path.abspath(os.path.join(os.path.join(os.getcwd(), '..'), '..'))  # 项目根目录
sys.path.append(project_path)  # 添加路径到系统路径中

default_model_path = f'{project_path}/models'
default_dataset_path = f'{project_path}/datasets'

# working_loc = 'Lingjiang'
# saving_dir_dict = {'Lingjiang': 'E:/PhD_research/AI4Science/Working_dir/NeuralODE'}
# default_saving_path = f'{saving_dir_dict[working_loc]}/ISO17'

################################################ 上面是一些环境设定 #######################################################

from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.datasets as datasets  # 数据集
import torchvision.transforms as transforms  # 对数据进行处理的库
import torch.utils.data as data_utils  # 数据处理工具库

class Model(nn.Module):
    def __init__(self, num_feature, num_class):
        super(Model, self).__init__()
        self.predict = nn.Linear(num_feature, num_class)

    def forward(self,x):
        out = self.predict(x)
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # FC - Fully Connected layer
        self.fc = torch.nn.Linear(14*14*32, 10)  # 0-9的概率分布，所以是10维的

    def forward(self,x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)  # 将数据展平，变成一个向量
        out = self.fc(out)  # FC层的输入一定是一个向量
        return out


if __name__ == '__main__':
    num_feature = 100
    num_class = 10

    batch_size = 64

    learning_rate = 0.001  # 学习率

    # 设备API
    use_cuda = torch.cuda.is_available()  # 判断是否用GPU加速

    # 创建数据集
    train_data = datasets.MNIST(root=default_dataset_path, train=True, transform=transforms.ToTensor(), download=False)  # 导入训练集
    test_data = datasets.MNIST(root=default_dataset_path, train=False, transform=transforms.ToTensor(), download=False)  # 导入测试集
    print(len(train_data), len(test_data))

    # 创建 mini batch，防止一次性加载数据集到内存中，导致内存溢出
    train_loader = data_utils.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)  # 创建训练集的 mini batch (打乱数据集可以丰富数据集)
    test_loader = data_utils.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Define neural network model (创建模型)
    # model = Model(num_feature, num_class)
    model = CNN()

    if use_cuda:
        model = model.cuda()  # 将模型部署在CUDA上

    # 定义损失函数
    # loss_func = torch.nn.MSELoss()  # 回归问题：这里是均方误差损失函数
    loss_func = torch.nn.CrossEntropyLoss()  # 分类问题：这里是交叉熵损失函数

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器，这里是Adam优化器

    # Training
    # for epoch in tqdm(range(10), desc='Training process'):
        # 这里应该有训练代码
        # ...

    # 针对 mini batch 的训练
    for epoch in range(10):

        # 每个 epoch 训练一个 mini batch
        for i, (images, labels) in enumerate(train_loader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            loss = loss_func(outputs, labels)  # 传入预测值和真实值, 计算损失

            optimizer.zero_grad()  # 清空梯度，防止累加
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()   # 更新参数

            # print(f'epoch - {epoch+1}, iter - {i}], loss: {loss.item():.4f}')  # 监控训练过程

        # 每个epoch结束后，对模型进行评估 (Test/Eval)
        loss_test, accuracy = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)  # outputs = batchsize * num_class
            loss_test += loss_func(outputs, labels)

            _, pred = outputs.max(1)  # 选择第一个维度
            accuracy += (pred == labels).sum().item()  # 计算预测正确的数量

        loss_test = loss_test / (len(test_data)//batch_size)  # 除以batch总量，计算损失
        accuracy = accuracy / len(test_data)  # 除以样品总量，计算准确率

        print(f'epoch - {epoch + 1}, accuracy - {accuracy}], test loss: {loss_test.item():.4f}')  # 监控训练过程


    # pass