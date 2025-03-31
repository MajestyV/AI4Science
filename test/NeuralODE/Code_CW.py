# 导入依赖
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 避免多个 OpenMP 运行时库实例的问题。

import math
import numpy as np

from typing import Union, Callable
# from IPython.display import clear_output
# from tqdm.notebook import tqdm
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
# matplotlib inline
import seaborn as sns
sns.color_palette("bright")
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F
from torch.autograd import Variable


def ode_solve(
        z0: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor,
        f: torch.nn.Module, h_max: float = 0.05, verbose: bool = False
):
    """
    常微分方程数值解法之欧拉法

    z0: hidden states 初值;
    t0: 初始时刻;
    t1: 终止时刻;
    f:  微分方程 dz/dt = f, 这里是神经网络;
    h_max: 预设的将 [t0, t1] 分割成细小的离散区间的间隔, 用于计算解微分方程的步数, 而实际的每个区间间隔需要根据步数进一步计算;
    verbose: 是否输出日志信息
    """

    # 根据 起始与终止时刻的绝对数值差 abs(t1 - t0) 以及 预设的离散区间间隔 h_max 来计算需要迭代的步数
    n_steps = math.ceil((abs(t1 - t0) / h_max).max().item())
    # 再由步数进一步计算出实际的离散区间间隔
    h = (t1 - t0) / n_steps

    t = t0
    z = z0
    # 用欧拉法迭代计算
    for _ in range(n_steps):
        z = z + h * f(z, t)
        t = t + h

    if verbose:
        print(f"ODE solved from time {t0} to {t1} with total {n_steps} steps.")

    return z


class ODEF(nn.Module):
    """
    A superclass of parameterized dynamics function: f = dz/dt.
    Moreover, this class implement a method that computes the augmented dynamics: f_aug,
    it depends on the gradient of the function wrt. its inputs & parameters: df/dz, df/dp df/dt.
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclass of ODEF must implement this method.")

    def forward_with_grad(self, z, t, grad_outputs):
        """
        Compute f = dz/dt and vJps: a * df/dz, a * df/dp, a * df/dt.
        It will be invoked in backward process for continous adjoint sensitivity method.
        """

        a = grad_outputs
        batch_size = z.shape[0]

        # f = dz/dt
        out = self.forward(z, t)

        # Compute vector Jocabian products(vJps)
        # a(z) * df/dz, a(t) * df/dt, a(p) * df/dp
        # 计算 vJps: 即 `a` 的转置分别与 `out` 相对于 `z`, `t` 以及 `self.parameters()` 的导数相乘
        # 由于该方法(`forward_with_grad()`)在一次 backward() 中会被多次调用
        # 因此保险起见，这里设置 `retain_graph=True` 来维护计算图而保持非叶子节点的梯度
        # 但其实该方法会在 `torch.set_grad_enabled(True)` 上下文管理器下调用,
        # 而这里有进行了前向过程 `out = self.forward(z, t)`, 所以会自动创建计算图
        # (亲测不设置 `retain_graph=True` 也 work)
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )

        # 以下需要扩充多一个 batch size 的维度，以便于它们能够和 dz/dt = f & a(z) * df/dz concat 起来
        if adfdp is not None:
            # (1, *)
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            # (batch_size, *)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            # (batch_size, 1)
            adfdt = adfdt.expand(batch_size, 1) / batch_size

        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        """ Flatten all parameters and concat them. """

        flat_parameters = []
        for p in self.parameters():
            flat_parameters.append(p.flatten())

        return torch.cat(flat_parameters)


class ODEAdjoint(torch.autograd.Function):
    """
    封装了前向与反向 ODE 的求解过程，并且在求解反向 ODE 中利用 adjoint method 来计算梯度
    Incapsulates forward & backward(with continuous adjoint sensitivity method) passes for Neural ODE.
    Instead of using torch.nn.Module, we use torch.autograd.Function here for implementing custom backward function.
    """

    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func: Union[nn.Module, Callable]):
        bs, *z_shape = z0.size()
        # 离散时刻(t0, t1, t2, ...)的数量
        time_len = t.size(0)

        # NOTE: We don't need to store the gradients here,
        # because we can compute gradients in backward function with continous adjoint sensitivity method.
        # 注意这里的前向过程不需要梯度, 因为 adjoint method 会在自定义的反向传播里通过解反向 ODE 将梯度计算出来
        with torch.no_grad():
            # We will return multiple hidden states for each discreted time,
            # loss would depend on each of them.
            # 返回每个离散时间点对应(解出)的 hidden states, 它们都会直接去计算 loss
            # 也就是 loss 不仅仅与最终的输出相关, 还与中间状态相关
            z = torch.zeros(time_len, bs, *z_shape).to(z0)

            z[0] = z0
            for t_i in range(time_len - 1):
                # 数值求解前向 ODE, z_{t_{i + 1}} = z_{t_{i}} + \int_{t_{i}}^{t_{i + 1}} f dt
                z0 = ode_solve(z0, t[t_i], t[t_i + 1], func)
                z[t_i + 1] = z0

        # ctx 用于保存反向传播时要用到的一些量
        ctx.func = func
        # 这里 clone() 是避免前向 ODE 解出来的 hidden states 受到外部影响,
        # 因为它们后续要被用作反向 ODE 的初值
        # ps: clone() 返回的 tensor 与原 tensor 不共享内存，也是就两者的修改互不影响,
        # clone 出来的 tensor 本身不会存储梯度, 因其在计算图中不是叶子节点而是中间节点, 但其梯度会叠加到原 tensor 上
        ctx.save_for_backward(t, z.clone(), flat_parameters)

        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        Custom backward pass, for calculating gradients with continous adjoint method.
        NOTE: In forward function, we output intermediate hidden states for loss computation,
        thus here will receive multiple gradients for each hidden state that `dLdz` shape: (time_len, batch_size, *z_shape)
        """

        # f = dz/dt
        func = ctx.func
        # (time_len,); (time_len, bs, *z_shape); (n_params,)
        t, z, flat_parameters = ctx.saved_tensors

        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Flatten dL/dz here for convenience.
        # 将最后一维展平 *z_shape -> n_dim
        dLdz = dLdz.view(time_len, bs, n_dim)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            建模 augmented 动态系统的微分方程
            tensors here are temporal slices
            t_i - is tensor with size: (1,)
            aug_z_i - is tensor with size: (bs, n_dim + n_dim + n_params + 1)
            """

            # z_{i}, a(z_{i})
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2 * n_dim]
            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)

            with torch.set_grad_enabled(True):
                # 将 tensor 从计算图中剥离出来(`deteach()`)并设置为接受梯度,
                # 因为这里过来的 t_i 和 z_i 可能并非是计算图的叶子节点(因而没有梯度),
                # 比如 z_i 就是由前向过程中 clone() 而来的
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)

                # Outputs evaluated result and vJps
                # 通过让神经网络做一次前向过程得到: f_{i} = dz_{i}/dt_{i};
                # 然后, 通过 torch.autograd 自动微分框架计算3个 vJps:
                # a(z_{i}) * df_{i}/dz_{i}, a(z_{i}) * df_{i}/dt_{i}, a(z_{i}) * df_{i}/dp
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)

                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # 展平最后一维 *z_shape -> n_dim
            # 主要是外面的 z 和 a(z) 都做了这样的操作, 于是这里就是为了维度能够对应起来
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)

            # NOTE: da(z)/dt = -a(z) * df/dz, da(p)/dt = -a(z) * df/dp, da(t)/dt = -a(z) * df/dt
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        # NOTE: 这整个大过程不需要梯度, 仅在计算 vJps 的时候 enable 梯度
        with torch.no_grad():
            # 为 hidden states, 网络参数 以及 时间变量 创建对应的伴随状态
            # NOTE: `adj_p` & `adj_t` 多了 `bs` 这一维度(按理说梯度应与张量的 shape 一致),
            # 为的是能够和 `adj_z` 拼起来作为 augmented states.
            # 多出来的这一维不要紧，最终会在这个维度上 sum 起来对应回张量的维度

            # a(z)
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            # a(p)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p, we need to return gradients for all discreted times a(t0), a(t1), ..., a(T)
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            # 依次求解每个离散区间的反向 ODE
            # From discreted step T to step 1(step 1 will result to step 0).
            for i_t in range(time_len - 1, 0, -1):
                # z_{i}
                z_i = z[i_t]
                # t_{i}
                t_i = t[i_t]
                # dz_{i}/dt_{i} (bs, *z_shape) -> (bs, n_dim)
                f_i = func(z_i, t_i).view(bs, n_dim)

                ''' Compute direct gradients (dL/dz; dL/dt = dL/dz * dz/dt) '''

                # dL/dz_{i} (bs, n_dim)
                dLdz_i = dLdz[i_t]
                # 利用链式求导法则算出 loss 相对于时间变量的直接梯度
                # dL/dt_{i} (bs, 1)
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                ''' Adjust adjoints with direct gradients '''

                # 根据 loss 过来的直接梯度对伴随状态进行调整
                adj_z += dLdz_i
                # This quantity is for computing dL/dt_{i - 1}, which intuitively, has the opposite gradient direction compared to t_{i},
                # because t_{i - 1} shortens the integration time interval when t_{i - 1} is increased.
                # Therefore, the initial value for this gradient, is the negative of dL/dt_{i}.
                # For details, pls reference: https://github.com/rtqichen/torchdiffeq/issues/218
                # 注意，这里的初值是负的 dLdt
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # 将 hidden states 以及各伴随状态拼在一起作为 augmented 状态输入到 ODE 求解器中
                # Pack augmented variable: z_{i}, a(z_{i}), a(p_{i}), a(t_{i})
                # (bs, n_dim + n_dim + n_params + 1)
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)
                # 将多个状态(`aug_z`)和对应的微分方程组合在一起(`augmented_dynamics`)后,
                # 就可以求解器同时解多个反向 ODE
                # Solve augmented system backwards
                # Get z_{i - 1}, a(z_{i - 1}), a(p_{i - 1}), a(t_{i - 1})
                aug_ans = ode_solve(aug_z, t_i, t[i_t - 1], augmented_dynamics)

                # 从解得的 augmented 状态中分离出各个伴随状态
                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2 * n_dim]
                # Note: we should accumulated sum all discreted adjoints for params
                adj_p[:] += aug_ans[:, 2 * n_dim:(2 * n_dim + n_params)]
                adj_t[i_t - 1] = aug_ans[:, (2 * n_dim + n_params):]

                del aug_z, aug_ans

            # 在 t0 时刻也有 loss 过来的直接梯度，于是最后这里也要调整一下
            # dL/dz_{0}
            dLdz_0 = dLdz[0]
            # dL/dt_{0}
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0

        # 输出的数量需要和前向过程中输入的数量对应起来
        # Gradients for each input of forward.
        # dL/dz, dL/dt for all discrete times, dL/dp, None for function call.
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None

class NeuralODE(nn.Module):
    def __init__(self, func: Union[nn.Module, Callable]):
        super(NeuralODE, self).__init__()
        # 参数化的微分方程，在这里就是神经网络
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence: bool = False):
        # 将 t 的 dtype & device 都变为与 z0 的一致
        t = t.to(z0)
        # ODEAdjoint 是 `torch.autograd.Function` 而非 `torch.nn.Module`, 这是为了能够实现自定义的反向传播
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)

        # `return_whole_sequence` 代表是否返回所有离散时刻的输出状态
        return z if return_whole_sequence else z[-1]

########################################################################################################################
class LinearODEF(ODEF):
    def __init__(self, W):
        super(LinearODEF, self).__init__()

        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        return self.lin(x)

class SpiralFunctionExample(LinearODEF):
    def __init__(self):
        super(SpiralFunctionExample, self).__init__(Tensor([[-0.1, -1.], [1., -0.1]]))


def to_np(x):
    return x.detach().cpu().numpy()


def plot_trajectories(obs=None, times=None, trajs=None, save=None, figsize=(16, 8)):
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

class RandomLinearODEF(LinearODEF):
    def __init__(self):
        super(RandomLinearODEF, self).__init__(torch.randn(2, 2) / 2.)

def conduct_experiment(ode_true: nn.Module, ode_trained: nn.Module, n_steps: int, name: str, plot_freq: int = 10):
    device = next(ode_true.parameters()).device

    # 轨迹的起始点
    # (1, 2)
    z0 = Variable(torch.Tensor([[0.6, 0.3]])).to(device)

    t_max = 6.29 * 5
    n_points = 200

    # index
    index_np = np.arange(0, n_points, 1, dtype=np.int32)
    # (n_points, 1)
    index_np = np.hstack([index_np[:, None]])

    # time piont
    times_np = np.linspace(0, t_max, num=n_points)
    # (n_points, 1)
    times_np = np.hstack([times_np[:, None]])

    # (n_points, 1, 1) 3 个维度分别对应时间序列长度(`time_len`), `batch size`, 向量编码维度
    times = torch.from_numpy(times_np[:, :, None]).to(z0)
    print(f"times shape: {times.shape}")

    # Solve out the truth trajectory.
    # 利用真实的 ODE 计算出所有时间点的状态 z(所有离散时刻所对应的点的位置)，它们连起来构成了目标轨迹
    # (n_points, 1, 2)
    obs = ode_true(z0, times, return_whole_sequence=True).detach()
    # 为了更具真实性，于是加上一点噪声(增添生活中的随机性)
    obs = obs + torch.randn_like(obs) * 0.01

    # Get trajectory of random timespan
    # 一个 batch 中，时间序列长度的上下限分别是 `min_delta_time` 和 `max_delta_time`
    min_delta_time = 1.0
    max_delta_time = 5.0
    # 一个 batch 最多包含的离散时间点数量
    max_points_num = 32

    def create_batch():
        """ 生成一个 batch 的 data: 对轨迹进行采样，只截取完整轨迹的一部分 """

        t0 = np.random.uniform(0, t_max - max_delta_time)
        t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)
        idx = sorted(np.random.permutation(index_np[(times_np > t0) & (times_np < t1)])[:max_points_num])

        # 选出轨迹点和对应的时间点
        obs_ = obs[idx]
        ts_ = times[idx]

        return obs_, ts_

    optimizer = torch.optim.Adam(ode_trained.parameters(), lr=0.01)

    # 迭代一定步数来训练网络
    for i in tqdm(range(n_steps), desc="Training Neural ODE"):
        obs_, ts_ = create_batch()
        # NOTE: 只需取第一个轨迹点送入网络中让其预测该 batch 中剩下的所有轨迹点
        z_ = ode_trained(obs_[0], ts_, return_whole_sequence=True)
        # 然后与真实的轨迹点对比算出 loss
        loss = F.mse_loss(z_, obs_.detach())

        optimizer.zero_grad()
        # 由于 loss 涉及到多个时间点的输出, 反向传播路径中它们可能共享一些中间的计算图节点,
        # 并且在 ODEAdjoint 中自定义的 backward() 方法里涉及到多次求梯度(计算 vJp 那部分),
        # 因此保险起见，这里用了 `retain_graph=True`, 避免某条路径计算完后释放掉中间节点而影响其它路径
        # (亲测不设置 `retain_graph=True` 也 work)
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % plot_freq == 0:
            # 训练一段时间后，让网络去预测整条完整的轨迹(而非仅仅是一个 batch 的轨迹)
            z_p = ode_trained(z0, times, return_whole_sequence=True)
            plot_trajectories(obs=[obs], times=[times], trajs=[z_p], save=f"E:/PhD_research/AI4Science/Working_dir/NeuralODE/{name}_{i}.png")
            # clear_output(wait=True)

            plt.close()

if __name__ == '__main__':
    # dev = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 单GPU推荐使用这行代码

    ode_true = NeuralODE(SpiralFunctionExample()).to(device)
    ode_trained = NeuralODE(RandomLinearODEF()).to(device)

    conduct_experiment(ode_true, ode_trained, 700, "linear")