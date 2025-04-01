import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 避免多个 OpenMP 运行时库实例的问题

import sys
project_path = os.path.abspath(os.path.join(os.path.join(os.getcwd(), '..'), '..'))  # 项目根目录
sys.path.append(project_path)  # 添加路径到系统路径中

default_model_path = f'{project_path}/models'

working_loc = 'Lingjiang'
saving_dir_dict = {'Lingjiang': 'E:/PhD_research/AI4Science/Working_dir/NeuralODE'}
default_saving_path = f'{saving_dir_dict[working_loc]}/SeqGeneration'

################################################ 上面是一些环境设定 #######################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm  # 进度条可视化模块

# 导入pytorch环境
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable

from src import to_np, NeuralODE

# 定义一个只包含线性层的神经网络来充当ODE函数，要继承ODE_func类，来获取计算vJp的能力
class Linear_ODEF(NeuralODE.ODE_func):
    def __init__(self, W):
        super(Linear_ODEF, self).__init__()

        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)  # 将W作为参数传入，这样就可以通过反向传播来更新W，进行学习

    def forward(self, x, t):
        return self.lin(x)


def gen_batch(batch_size, n_sample=100):
    """
    以固定的 batch size 选取一定数量的轨迹并且随机选取每条轨迹中的部分序列
    Generate a batch of samples.
        Args:
            n_sample (int): Number of time steps.
        Returns:
            Sampled trajectories (Tensor): n_sample x batch_size x 2;
            Sampled time steps (Tensor): n_sample x batch_size x 1.
    """

    time_len = samp_trajs.shape[0]
    n_sample = min(n_sample, time_len)

    n_batches = samp_trajs.shape[1] // batch_size
    for i in range(n_batches):
        if n_sample > 0:
            # (time_len - n_sample,) one-hot liked.
            t0_idx = np.random.multinomial(1, [1. / (time_len - n_sample)] * (time_len - n_sample))
            # t0_idx 的取值范围是 [0, time_len - n_sample - 1]
            t0_idx = np.argmax(t0_idx)
            tM_idx = t0_idx + n_sample
        else:
            t0_idx = 0
            tM_idx = time_len

        frm, to = batch_size * i, batch_size * (i + 1)
        # 使用 `yeild` 达到按需取用的效果，避免提前将数据实例化出来占用内存
        yield samp_trajs[t0_idx:tM_idx, frm:to], samp_ts[t0_idx:tM_idx, frm:to]

if __name__ == '__main__':
    ################################################### 一些全局设定 #####################################################
    model_name = 'ODE-VAE'

    learning_rate = 1e-3  # 优化器的学习率

    batch_size = 30
    num_epochs = 600  #

    dump = 50  #

    ################################################### 数据生成模块 #####################################################
    print("Generating dataset...")

    # Number of discrete time
    n_points = 200
    # Number of different spirals
    num_spirals = 300

    # index
    index_np = np.arange(0, n_points, 1, dtype='int')
    # (n_points, 1)
    index_np = np.hstack([index_np[:, None]])

    t_max = 6.29 * 5

    # time point
    times_np = np.linspace(0, t_max, num=n_points)
    # (n_points, num_spirals)
    times_np = np.hstack([times_np[:, None]] * num_spirals)
    # (n_points, num_spirals, 1)
    times = torch.from_numpy(times_np[:, :, None]).to(torch.float32)

    # 均值为0, 标准差为1的正态分布
    normal01 = torch.distributions.Normal(0, 1.0)

    # 为螺旋轨迹采样参数, 每条螺旋轨迹的微分方程由一个二维矩阵决定
    W11 = -0.1 * normal01.sample((num_spirals,)).abs() - 0.05
    W22 = -0.1 * normal01.sample((num_spirals,)).abs() - 0.05
    W21 = -1.0 * normal01.sample((num_spirals,)).abs()
    W12 = 1.0 * normal01.sample((num_spirals,)).abs()

    # 轨迹的初始點
    # (num_spirals, 2)
    x0 = Variable(normal01.sample((num_spirals, 2))) * 2.0

    # 收集所有轨迹
    xs_list = []
    for i in range(num_spirals):
        # 为了交替生成顺时针和逆时针的螺旋轨迹
        if i % 2 == 1:
            W21, W12 = W12, W21

        func = Linear_ODEF(Tensor([[W11[i], W12[i]], [W21[i], W22[i]]]))
        ode = NeuralODE.NeuralODE(func)

        # 利用 ODE 生成轨迹上的每个点
        # (n_points, 1, 2)
        xs = ode(x0[i:(i + 1)], times[:, i:(i + 1)], return_whole_sequence=True)
        xs_list.append(xs)

    noise_std = 0.02

    # (n_points, num_spirals, 2)
    orig_trajs = torch.cat(xs_list, dim=1).detach()
    # 为了更具备真实性, 所以在原始轨迹上加些噪声后作为训练样本
    samp_trajs = orig_trajs + torch.randn_like(orig_trajs) * noise_std
    # (n_points, num_spirals, 1)
    samp_ts = times

    ################################################# 模型训练及轨迹生成 ##################################################
    use_cuda = torch.cuda.is_available()  # 判断是否用GPU加速

    model = NeuralODE.ODE_VAE(output_dim=2, hidden_dim=64, latent_dim=6)  # 调用模型
    if use_cuda:
        model.cuda()  # 将模型部署在CUDA上

    device = next(ODE_NN.parameters()).device  # 查看计算设备并确认
    print(f'Use CUDA: {use_cuda} - Device used: {device}')  # 记录CUDA的使用情况

    optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=learning_rate)  # 优化器，这里选用Adam

    preload = False
    if preload:
        model.load_state_dict(torch.load(f'{default_model_path}/{model_name}.sd'))


    batch_mean_loss, batch_median_loss = ([],[])  # 创建两个列表用以记录训练batch的均值以及中位数loss

    for epoch_idx in tqdm(range(num_epochs), desc='Training LatentODE'):
        losses = []
        train_iter = gen_batch(batch_size)

        for x, t in train_iter:
            if use_cuda:
                # (n_sample_t, batch_size, 2); (n_sample_t, batch_size, 1)
                x, t = x.to(device), t.to(device)

            ''' 不规则采样 '''
            # Max time length
            max_len = np.random.choice([30, 50, 100])
            # Select indices
            permutation = np.random.permutation(t.shape[0])
            np.random.shuffle(permutation)
            # Sort indices
            permutation = np.sort(permutation[:max_len])

            optim.zero_grad()

            # (max_len, batch_size, 2); (max_len, batch_size, 1)
            x, t = x[permutation], t[permutation]
            # (max_len, batch_size, 2); (batch_size, latent_dim); (batch_size, latent_dim); (batch_size, latent_dim)
            x_p, z, z_mean, z_log_var = model(x, t)

            # VAE loss 包括 reconstruction loss & KL-Divergence
            # (batch_size,)
            kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)
            # 根据最小化正态分布的负对数似然所推出来的式子(但不计方差那部分的损失)
            # (batch_size,)
            reconstruction_loss = 0.5 * ((x - x_p) ** 2).sum(-1).sum(0) / noise_std ** 2
            loss = torch.mean(reconstruction_loss + kl_loss) / max_len
            losses.append(loss.item())

            loss.backward()
            optim.step()

        # print(f"Epoch {epoch_idx}")

        frm, to, to_seed = 0, 200, 50

        # 截取部分序列
        seed_trajs = samp_trajs[frm:to_seed]
        # 完整的时间序列
        ts = samp_ts[frm:to]
        if use_cuda:
            seed_trajs = seed_trajs.to(device)
            ts = ts.to(device)

        # 由截取的部分序列外推至整个序列
        # (to - frm, num_spirals, 2)
        samp_trajs_p = to_np(model.generate_with_seed(seed_trajs, ts))

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 9))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            # 截取的部分序列
            ax.scatter(to_np(seed_trajs[:, i, 0]), to_np(seed_trajs[:, i, 1]), c=to_np(ts[frm:to_seed, i, 0]),
                       cmap=cm.plasma)
            # 原始的轨迹
            ax.plot(to_np(orig_trajs[frm:to, i, 0]), to_np(orig_trajs[frm:to, i, 1]))
            # 模型外推出来的整条轨迹
            ax.plot(samp_trajs_p[:, i, 0], samp_trajs_p[:, i, 1])

        if (epoch_idx == 0) or ((epoch_idx+1) % dump == 0):
            plt.savefig(f'{default_saving_path}/{model_name}_Epoch-{epoch_idx}.png')

        # plt.show(block=True)
        plt.close()  # 关闭图像释放内存

        # 保存训练过程的loss，用于后续分析
        batch_mean_loss.append(np.mean(losses))
        batch_median_loss.append(np.median(losses))

    # 记录训练过程
    file = open(f'{default_saving_path}/{model_name}_loss.txt', 'w')  #将要输出保存的文件地址
    for i in range(num_epochs):
        file.write(f'Epoch {i}: Mean Loss - {batch_mean_loss[i]}, Median Loss - {batch_median_loss[i]}')
        file.write('\n')  # 换行
    file.close()

    torch.save(model.state_dict(), f'{default_model_path}/{model_name}.sd')  # 保存模型参数
    print(f'Checkpoint has been saved to: {default_model_path}/{model_name}.sd.')