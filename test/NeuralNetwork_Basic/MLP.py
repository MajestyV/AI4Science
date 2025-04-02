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

class Model(nn.Module):
    def __init__(self, num_feature, num_class):
        super(Model, self).__init__()
        self.predict = nn.Linear(num_feature, num_class)

    def forward(self,x):
        out = self.predict(x)
        return out

if __name__ == '__main__':
    num_feature = 100
    num_class = 10

    learning_rate = 0.001  # 学习率

    # 创建数据集
    train_data = datasets.MNIST(root=default_dataset_path, train=True, transform=transforms.ToTensor(), download=False)  # 导入训练集
    test_data = datasets.MNIST(root=default_dataset_path, train=False, transform=transforms.ToTensor(), download=False)  # 导入测试集

    print(len(train_data), len(test_data))

    exit()

    breakpoint()
    # download=True)

    # 创建模型
    model = Model(num_feature, num_class)

    loss = torch.nn.MSELoss()  # 定义损失函数，这里是均方误差损失函数

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器，这里是Adam优化器

    # Training
    # for epoch in tqdm(range(10), desc='Training process'):
        # 这里应该有训练代码
        # ...



    # pass