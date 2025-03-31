import torch
import torch.nn as nn

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
        # (time_len, bs, input_dim + 1)
        xt = torch.cat((x, t), dim=-1)

        # `flip()`: 在第一个维度上反转，也就是倒转时间序列
        # GRU 输出的第二个变量(也就是以下的 `h0`)的 shape 是 (num_layers * num_directions, bs, hidden_dim)
        _, h0 = self.rnn(xt.flip((0,)))
        # (bs, 2 * latent_dim)
        z0 = self.hid2lat(h0[0])

        # 编码结果是 latent 的均值与(对数)方差，VAE 的套路
        z0_mean = z0[:, :self.latent_dim]
        z0_log_var = z0[:, self.latent_dim:]

        return z0_mean, z0_log_var

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 这里的 RNN 用的是1层 GRU
        # 输入维度加1是因为会将时间向量(维度为1)拼接到输入中
        self.rnn = nn.GRU(input_dim + 1, hidden_dim)
        self.hid2lat = nn.Linear(hidden_dim, 2 * latent_dim)

    def forward(self, x, t):
        # Concatenate time to input
        t = t.clone()
        # 这里使用时间差是为了增强 RNN 对不规则采样时间序列的适应能力
        t[1:] = t[:-1] - t[1:]
        t[0] = 0.
        # (time_len, bs, input_dim + 1)
        xt = torch.cat((x, t), dim=-1)

        # `flip()`: 在第一个维度上反转，也就是倒转时间序列
        # GRU 输出的第二个变量(也就是以下的 `h0`)的 shape 是 (num_layers * num_directions, bs, hidden_dim)
        _, h0 = self.rnn(xt.flip((0,)))
        # (bs, 2 * latent_dim)
        z0 = self.hid2lat(h0[0])

        # 编码结果是 latent 的均值与(对数)方差，VAE 的套路
        z0_mean = z0[:, :self.latent_dim]
        z0_log_var = z0[:, self.latent_dim:]

        return z0_mean, z0_log_var