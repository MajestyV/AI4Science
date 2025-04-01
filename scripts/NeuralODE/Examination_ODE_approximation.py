import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 避免多个 OpenMP 运行时库实例的问题

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm  # 进度条可视化模块

from src import NeuralODE

# 导入pytorch环境
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F      # 常用的神经网络函数
from torch.autograd import Variable  # 自动求导模块

working_loc = 'Lingjiang'

saving_dir_dict = {'Lingjiang': 'E:/PhD_research/AI4Science/Working_dir/NeuralODE'}

# 定义一个神经网络来充当ODE函数，要继承ODE_func类，来获取计算vJp的能力
class Linear_ODEF(NeuralODE.ODE_func):
    def __init__(self, W):
        super(Linear_ODEF, self).__init__()

        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)  # 将W作为参数传入，这样就可以通过反向传播来更新W，进行学习

    def forward(self, x, t):
        return self.lin(x)

class Example_spiral(Linear_ODEF):
    ''' Toy model 数据集生成类 '''
    def __init__(self):
        super(Example_spiral, self).__init__(Tensor([[-0.1, -1.], [1., -0.1]]))

class Random_ODEF(Linear_ODEF):
    ''' 随机初始化的二维矩阵作为要学习的参数 '''
    def __init__(self):
        super(Random_ODEF, self).__init__(torch.randn(2, 2) / 2.)

# 此函数可以将torch.Tensor转换为numpy.ndarray
def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def plot_trajectories(obs=None, times=None, trajs=None, save=None, figsize=(16, 8)):
    ''' 此函数专用于可视化轨迹 '''

    plt.figure(figsize=figsize)

    # NOTE: 传进来的 obs, times, trajs 都是 list，代表可能要画多个轨迹(本例子只有一个条轨迹，因此 list 长度为1)

    if obs is not None:
        if times is None:
            times = [None] * len(obs)

        for o, t in zip(obs, times):
            o, t = to_np(o), to_np(t)
            for b_i in range(o.shape[1]):
                plt.scatter(o[:, b_i, 0], o[:, b_i, 1], c=t[:, b_i, 0], cmap=cm.plasma)

    if trajs is not None:
        for z in trajs:
            z = to_np(z)
            plt.plot(z[:, 0, 0], z[:, 0, 1], lw=1.5)

        if save is not None:
            dir_name = os.path.split(save)[0]
            os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save)

if __name__ == '__main__':
    examination_title = 'ODE_approximation'

    # 选择计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 单卡 (单GPU) 推荐使用这行代码

    ODE_truth = NeuralODE.NeuralODE(func=Example_spiral()).to(device)  # 目标ODE
    ODE_NN = NeuralODE.NeuralODE(func=Random_ODEF()).to(device)        # 随机初始化的神经网络ODE

    num_steps = 700  # 训练迭代次数
    plot_freq = 10   # 每隔多少步画一次图

    use_cuda = torch.cuda.is_available()  # 判断是否用GPU加速
    print(f"Use CUDA: {use_cuda}")        # 记录CUDA的使用情况

    device = next(ODE_NN.parameters()).device
    print(f"Device: {device}")            # 记录CUDA的使用情况

    # 轨迹的起始点: (1, 2)
    z0 = Variable(torch.Tensor([[0.6, 0.3]])).to(device)

    t_max = 6.29 * 5
    n_points = 200

    index_np = np.arange(0, n_points, 1, dtype=np.int32)  # index
    index_np = np.hstack([index_np[:, None]])  # (n_points, 1)


    times_np = np.linspace(0, t_max, num=n_points)  # time piont
    times_np = np.hstack([times_np[:, None]])       # (n_points, 1)

    # (n_points, 1, 1) 3 个维度分别对应时间序列长度(`time_len`), `batch size`, 向量编码维度
    times = torch.from_numpy(times_np[:, :, None]).to(z0)
    print(f"times shape: {times.shape}")

    # 利用真实的 ODE 计算出所有时间点的状态 z(所有离散时刻所对应的点的位置)，它们连起来构成了目标轨迹 (Solve out the truth trajectory)
    # (n_points, 1, 2)
    obs = ODE_truth(z0, times, return_whole_sequence=True ).detach()
    obs = obs + torch.randn_like(obs) * 0.01  # 为了更具真实性，于是加上一点噪声(增添生活中的随机性)
    # print(f"obs shape: {obs.shape}")

    # Get trajectory of random timespan
    min_delta_time = 1.0  # 一个 batch 中，时间序列长度的上限，时间步间隔的下限
    max_delta_time = 5.0  # 一个 batch 中，时间序列长度的下限，时间步间隔的上限
    max_points_num = 32   # 一个 batch 最多包含的离散时间点数量

    def create_batch():
        """ 生成一个 batch 的 data: 对轨迹进行采样，只截取完整轨迹的一部分 """

        t0 = np.random.uniform(0, t_max - max_delta_time)
        t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)
        idx = sorted(np.random.permutation(index_np[(times_np > t0) & (times_np < t1)])[:max_points_num])

        # 选出轨迹点和对应的时间点
        obs_ = obs[idx]
        ts_ = times[idx]

        return obs_, ts_

    optimizer = torch.optim.Adam(ODE_NN.parameters(), lr=0.01)  # 优化器，这里选用Adam

    # 迭代一定步数来训练网络
    for i in tqdm(range(num_steps), desc="Training NeuralODE"):
        obs_, ts_ = create_batch()
        z_ = ODE_NN(obs_[0], ts_, return_whole_sequence=True)  # NOTE: 只需取第一个轨迹点送入网络中让其预测该 batch 中剩下的所有轨迹点
        loss = F.mse_loss(z_, obs_.detach())  # 然后与真实的轨迹点对比算出 loss， 这里用的是均方误差损失

        optimizer.zero_grad()
        # 由于 loss 涉及到多个时间点的输出, 反向传播路径中它们可能共享一些中间的计算图节点,
        # 并且在 ODEAdjoint 中自定义的 backward() 方法里涉及到多次求梯度(计算 vJp 那部分),
        # 因此保险起见，这里用了 `retain_graph=True`, 避免某条路径计算完后释放掉中间节点而影响其它路径
        # (亲测不设置 `retain_graph=True` 也 work)
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % plot_freq == 0:
            # 训练一段时间后，让网络去预测整条完整的轨迹(而非仅仅是一个 batch 的轨迹)
            z_p = ODE_NN(z0, times, return_whole_sequence=True)
            plot_trajectories(obs=[obs], times=[times], trajs=[z_p], save=f'{saving_dir_dict[working_loc]}/{examination_title}_{i}.png')

            plt.close()  # 关闭图像释放内存