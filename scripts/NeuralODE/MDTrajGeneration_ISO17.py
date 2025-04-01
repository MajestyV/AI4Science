import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 避免多个 OpenMP 运行时库实例的问题

import sys
project_path = os.path.abspath(os.path.join(os.path.join(os.getcwd(), '..'), '..'))  # 项目根目录
sys.path.append(project_path)  # 添加路径到系统路径中

default_model_path = f'{project_path}/models'
default_dataset_path = f'{project_path}/datasets/ISO17'

working_loc = 'Lingjiang'
saving_dir_dict = {'Lingjiang': 'E:/PhD_research/AI4Science/Working_dir/NeuralODE'}
default_saving_path = f'{saving_dir_dict[working_loc]}/ISO17'

################################################ 上面是一些环境设定 #######################################################
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条可视化模块

# 导入pytorch环境
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable

from src import to_np, ConfigSimilarity_MSE, NeuralODE  # 导入自定义模型

if __name__ == '__main__':
    ################################################### 一些全局设定 #####################################################
    model_name = 'ODE-VAE_ISO17'
    dataset_name = 'MDTraj_nstep-5000.npy'

    learning_rate = 1e-3  # 优化器的学习率

    num_epochs = 500  #

    dump = 50  #  保存模型训练结果的间隔

    preload = False  # 是否加载预训练模型

    ################################################### 数据导入模块 #####################################################
    print("Generating dataset...")

    MD_traj = np.load(f'{default_dataset_path}/{dataset_name}')  # 读取数据
    MD_traj = torch.from_numpy(MD_traj).to(torch.float32)  # 将数据转换为torch.Tensor格式

    # time = torch.from_numpy(np.linspace(0,))  # 生成时间序列，并转换为torch.Tensor格式

    # time point
    # times_np = np.linspace(0, t_max, num=n_points)
    # (n_points, num_spirals)
    # times_np = np.hstack([times_np[:, None]] * num_spirals)
    # (n_points, num_spirals, 1)
    # times = torch.from_numpy(times_np[:, :, None]).to(torch.float32)

    # 因为后面有串接操作，所以这里需要将数据扩展到跟MD_traj一样的维度
    # time_origin = np.linspace(0, 1, MD_traj.shape[0])  # 生成时间序列
    time_origin = np.array([i for i in range(MD_traj.shape[0])])  # 生成时间序列
    time = np.hstack([time_origin[:, None]] * MD_traj.shape[1])
    time = torch.from_numpy(time[:, :, None]).to(torch.float32)  # 将数据转换为torch.Tensor格式

    l_dataset = MD_traj.shape[0]  # 轨迹的长度

    # print(time.shape)

    # print(time)

    ''' 不规则采样 '''
    # max_len = np.random.choice([30, 50, 100])  # Max time length
    max_len = 3000 # Max time length
    permutation = np.random.permutation(time.shape[0])  # Select indices
    np.random.shuffle(permutation)
    permutation = np.sort(permutation[:max_len])  # Sort indices

    time_sampled = time[permutation]  # 根据采样的时间序列选取对应的轨迹
    MD_traj_sampled = MD_traj[permutation]  # 根据采样的时间序列选取对应的轨迹

    ################################################# 模型训练及轨迹生成 ##################################################
    use_cuda = torch.cuda.is_available()  # 判断是否用GPU加速

    model = NeuralODE.ODE_VAE(output_dim=3, hidden_dim=64, latent_dim=6)  # 调用模型
    if use_cuda:
        model.cuda()  # 将模型部署在CUDA上

    device = next(model.parameters()).device  # 查看计算设备并确认
    print(f'Use CUDA: {use_cuda} - Device used: {device}')  # 记录CUDA的使用情况

    optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=learning_rate)  # 优化器，这里选用Adam

    # 判断是否使用GPU加速
    if preload:
        model.load_state_dict(torch.load(f'{default_model_path}/{model_name}.sd'))

    batch_mean_loss, batch_median_loss = ([], [])  # 创建两个列表用以记录训练batch的均值以及中位数loss

    if use_cuda:
        # (n_sample_t, batch_size, 2); (n_sample_t, batch_size, 1)
        x, t = MD_traj_sampled.to(device), time_sampled.to(device)
    else:
        # (n_sample_t, batch_size, 2); (n_sample_t, batch_size, 1)
        x, t = MD_traj_sampled, time_sampled

    for epoch_idx in tqdm(range(num_epochs), desc='Training LatentODE'):
        losses = []

        optim.zero_grad()  # 清空梯度，防止累加

        # (max_len, batch_size, 2); (max_len, batch_size, 1)
        # x, t = x[permutation], t[permutation]
        # (max_len, batch_size, 2); (batch_size, latent_dim); (batch_size, latent_dim); (batch_size, latent_dim)
        x_p, z, z_mean, z_log_var = model(x, t)

        # VAE loss 包括 reconstruction loss & KL-Divergence
        # (batch_size,)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)
        # 根据最小化正态分布的负对数似然所推出来的式子(但不计方差那部分的损失)
        # (batch_size,)
        reconstruction_loss = 0.5 * ((x - x_p) ** 2).sum(-1).sum(0)
        loss = torch.mean(reconstruction_loss + kl_loss) / max_len
        losses.append(loss.item())

        loss.backward()
        optim.step()

        # print(f"Epoch {epoch_idx}")

        frm, to, to_seed = 0, 5000, 3000

        # 截取部分序列
        seed_trajs = MD_traj_sampled[frm:to_seed]
        # 完整的时间序列
        time_pred = time[frm:to]
        if use_cuda:
            seed_trajs = seed_trajs.to(device)
            time_pred = time_pred.to(device)

        # 由截取的部分序列外推至整个序列 [shape: (to - frm, num_spirals, 2)]
        samp_trajs_p = to_np(model.generate_with_seed(seed_trajs, time_pred))

        # print(samp_trajs_p.shape)

        if (epoch_idx == 0) or ((epoch_idx + 1) % dump == 0):  # 按照一定的间隔保存模型的输出
            model_predict = to_np(samp_trajs_p)
            np.save(f'{default_saving_path}/{model_name}_Epoch-{epoch_idx}.npy', model_predict)  # 保存模型的输出

            # 画图
            plt.figure(figsize=(10, 4))

            MSE_traj = [ConfigSimilarity_MSE(to_np(model_predict[i]), to_np(MD_traj[i])) for i in range(l_dataset)]

            plt.plot(time_origin, MSE_traj)

            plt.savefig(f'{default_saving_path}/{model_name}_Epoch-{epoch_idx}.png')

            # plt.show(block=True)
            plt.close()  # 关闭图像释放内存

        # 保存训练过程的loss，用于后续分析
        batch_mean_loss.append(np.mean(losses))
        batch_median_loss.append(np.median(losses))

    # 记录训练过程
    file = open(f'{default_saving_path}/{model_name}_loss.txt', 'w')  # 将要输出保存的文件地址
    for i in range(num_epochs):
        file.write(f'Epoch {i}: Mean Loss - {batch_mean_loss[i]}, Median Loss - {batch_median_loss[i]}')
        file.write('\n')  # 换行
    file.close()

    torch.save(model.state_dict(), f'{default_model_path}/{model_name}.sd')  # 保存模型参数
    print(f'Checkpoint has been saved to: {default_model_path}/{model_name}.sd.')