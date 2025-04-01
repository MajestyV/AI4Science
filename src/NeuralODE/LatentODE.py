import torch
import torch.nn as nn
from .NeuralODE_Basic import ODE_func, NeuralODE  # 直接调用 NeuralODE_Basic 中的 NeuralODE 基础模块

################################################### Encoder module #####################################################

class GRU_encoder(nn.Module):
    ''' GRU Encoder for LatentODE model '''
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers: int=1):
        super(GRU_encoder, self).__init__()

        # 解压缩参数，分别为输入维度、隐藏层维度、潜空间维度
        self.input_dim, self.hidden_dim, self.latent_dim = (input_dim, hidden_dim, latent_dim)

        self.gru = nn.GRU(input_dim+1, hidden_dim, num_layers)  # 输入维度加1是因为会将时间向量(维度为1)拼接到输入中
        self.hid2lat = nn.Linear(hidden_dim, 2*latent_dim)  # 线性层将隐藏状态投影到潜空间(维度乘2是因为要输出均值向量和对数方差向量)

    def forward(self, x, t):
        t = t.clone()  # Concatenate time to input
        # 这里使用时间差是为了增强 RNN 对不规则采样时间序列的适应能力
        t[1:] = t[:-1] - t[1:]
        t[0] = 0.

        xt = torch.cat((x, t), dim=-1)  # 将时间序列串接入输入，进行时序编码 [Shape：(time_len, bs, input_dim + 1)]

        # `flip()`: 在第一个维度上反转，也就是倒转时间序列
        # GRU 输出的第二个变量(也就是以下的 `h0`)的 shape 是 (num_layers * num_directions, bs, hidden_dim)
        _, h0 = self.gru(xt.flip((0,)))

        z0 = self.hid2lat(h0[0])  # 投影到潜空间 [Shape: (bs, 2 * latent_dim)]

        # 编码结果是 latent 的均值与(对数)方差，VAE 的套路
        z0_mean = z0[:, :self.latent_dim]
        z0_log_var = z0[:, self.latent_dim:]

        return z0_mean, z0_log_var

################################################### Decoder module #####################################################
class NN_ODEF(ODE_func):
    ''' 首先定义一个神经网络来充当ODE函数，要继承ODE_func类，来获取计算vJp的能力 '''
    def __init__(self, in_dim: int, hid_dim: int, time_invariant: bool = False):
        super(ODE_func, self).__init__()

        self.time_invariant = time_invariant
        if not time_invariant:
            in_dim += 1

        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)

        self.elu = nn.ELU(inplace=True)

    def forward(self, x, t):
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        h = self.elu(self.lin1(x))
        h = self.elu(self.lin2(h))
        out = self.lin3(h)

        return out

class NeuralODE_decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(NeuralODE_decoder, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 这里建模的 ODE 是 "time-invariant" 的，也就是不需要将时间向量输入到 Neural ODE 中
        func = NN_ODEF(latent_dim, hidden_dim, time_invariant=True)
        self.ode = NeuralODE(func=func)  # 定义NeuralODE

        self.l2h = nn.Linear(latent_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)

    def forward(self, z0, t):
        # Neural ODE 由初始 latent 预测出 latent 轨迹(整个序列的 latents)
        # (time_len, bs, latent_dim)
        zs = self.ode(z0, t, return_whole_sequence=True)
        # 对整个序列的 latents 进行解码，重构原输入序列
        hs = self.l2h(zs)
        xs = self.h2o(hs)

        return xs

############################################## Variational Auto-encoder ################################################

class ODE_VAE(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim, encoder: nn.Module=GRU_encoder,
                 decoder: nn.Module=NeuralODE_decoder):
        super(ODE_VAE, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = encoder(output_dim, hidden_dim, latent_dim)
        self.decoder = decoder(output_dim, hidden_dim, latent_dim)


    def forward(self, x, t, reparameterization_trick: bool=True):

        z_mean, z_log_var = self.encoder(x, t)  # 从Encoder的输出获取潜空间均值以及方差 [shape: (bs, latent_dim)]

        # 以重参数化手段代替直接采样，从而梯度能够传至 Encoder，进而得到训练，VAE 的套路 (若不使用重参数化采样，则直接使用均值)
        z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var) if reparameterization_trick else z_mean

        x_p = self.decoder(z, t)  # 从潜空间分布解码得到重构的序列 [shape: (time_len, bs, output_dim)]

        return x_p, z, z_mean, z_log_var

    @torch.no_grad()
    def generate_with_seed(self, seed_x, t):
        ''' 根据部分序列去进行序列生成 '''

        # 部分序列
        seed_t_len = seed_x.shape[0]
        # 根据部分序列去编码得到 latent
        z_mean, _ = self.encoder(seed_x, t[:seed_t_len])
        # 外推出完整的序列
        x_p = self.decoder(z_mean, t)

        return x_p