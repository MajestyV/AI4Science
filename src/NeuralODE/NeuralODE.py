# 导入基础库
import math
import numpy as np

# 导入pytorch环境
import torch
import torch.nn as nn
from torch import Tensor

from typing import Callable, Union  # 用于类型注解

def ODEsolver_Euler(z0: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor, f: torch.nn.Module, h_max: float = 0.05,
                    verbose: bool = False):
    """
    基于欧拉法的常微分方程 (Ordinary differential equation, ODE) 求解器
    :param z0: hidden states 初值
    :param t0: 初始时刻
    :param t1: 终止时刻
    :param f: 微分方程 dz/dt = f, 这里是神经网络
    :param h_max: 预设的将 [t0, t1] 分割成细小的离散区间的间隔, 用于计算解微分方程的步数, 而实际的每个区间间隔需要根据步数进一步计算
    :param verbose: 是否输出日志信息
    :return: 求解后的隐状态
    """

    # 根据起始与终止时刻的绝对数值差 abs(t1 - t0), 以及预设的离散区间间隔 h_max 来计算需要迭代的步数
    n_steps = math.ceil((abs(t1 - t0) / h_max).max().item())
    h = (t1 - t0) / n_steps  # 再由步数进一步计算出实际的离散区间间隔

    t, z = (t0, z0)  # 初始化时刻与隐状态

    # 用欧拉法迭代计算 ODE 终值
    for _ in range(n_steps):
        t = t + h
        z = z + h * f(z, t)

    if verbose:
        print(f"ODE solved from time {t0} to {t1} with total {n_steps} steps.")

    return z

class ODE_func(nn.Module):
    '''
    A superclass of parameterized function of dynamics: f = dz/dt.
    Moreover, this class implement a method that computes the augmented dynamics: f_aug, it depends on the gradient of
    the function with respect to. Its inputs & parameters: df/dz, df/dp df/dt.
    '''

    def forward(self, *args, **kwargs): raise NotImplementedError("Subclass of ODE_func must implement this method.")

    def forward_with_grad(self, z, t, grad_outputs):
        '''
        计算 f = dz/dt, 以及 vector Jocabian products (vJps): a * df/dz, a * df/dp, a * df/dt, 以实现自动微分。
        这对于后续的连续伴随灵敏度方法 (continous adjoint sensitivity method) 是至关重要的。
        '''

        a = grad_outputs          # 伴随状态 (Adjoint state) 向量
        batch_size = z.shape[0]   # 一次训练所选取的样本数

        out = self.forward(z, t)  # f = dz/dt

        # Compute vector Jocabian products (vJps): a(z) * df/dz, a(t) * df/dt, a(p) * df/dp
        # 计算 vJps: 即 `a` 的转置分别与 `out` 相对于 `z`, `t` 以及 `self.parameters()` 的导数相乘
        # 由于该方法(`forward_with_grad()`)在一次 backward() 中会被多次调用
        # 因此保险起见，这里设置 `retain_graph=True` 来维护计算图而保持非叶子节点的梯度
        # 但其实该方法会在 `torch.set_grad_enabled(True)` 上下文管理器下调用,
        # 而这里有进行了前向过程 `out = self.forward(z, t)`, 所以会自动创建计算图
        # (亲测不设置 `retain_graph=True` 也 work)
        # 变量前加*号可以解压可迭代对象，对应zip()压缩可迭代对象 (https://blog.csdn.net/weixin_40877427/article/details/82931899)
        adfdz, adfdt, *adfdp = torch.autograd.grad((out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
                                                   allow_unused=True, retain_graph=True)

        # 以下需要扩充多一个 batch size 的维度，以便于它们能够和 dz/dt = f & a(z) * df/dz concat 起来
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)  # (1, *)
            adfdp = adfdp.expand(batch_size, -1) / batch_size                       # (batch_size, *)
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size                        # (batch_size, 1)

        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        ''' Flatten all parameters and concat them. '''
        flat_parameters = []
        for p in self.parameters():
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

class ODE_adjoint(torch.autograd.Function):
    ''' 封装了前向与反向 ODE 的求解过程，并且在求解反向 ODE 中利用 adjoint method 来计算梯度 '''

    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func: Union[nn.Module, Callable], ODEsolver: Callable = ODEsolver_Euler):
        '''
        :param ctx: context, 上下文管理环境 (专用于静态方法中，替代self，详见：https://blog.csdn.net/littlehaes/article/details/103828130)
        '''

        batch_size, *z_shape = z0.size()
        time_len = t.size(0)  # 离散时刻(t0, t1, t2, ...)的数量

        # 注意这里的前向过程不需要储存梯度, 因为 adjoint method 会在自定义的反向传播里通过解反向 ODE 将梯度计算出来
        with torch.no_grad():
            # 返回每个离散时间点对应(解出)的 hidden states, 它们都会直接去计算 loss
            # 也就是 loss 不仅仅与最终的输出相关, 还与中间状态相关
            z = torch.zeros(time_len, batch_size, *z_shape).to(z0)

            z[0] = z0
            for t_i in range(time_len - 1):
                # 数值求解前向 ODE, z_{t_{i + 1}} = z_{t_{i}} + \int_{t_{i}}^{t_{i + 1}} f dt
                z0 = ODEsolver(z0, t[t_i], t[t_i + 1], func)
                z[t_i + 1] = z0

        ctx.func = func  # ctx 用于保存反向传播时要用到的一些量

        # 这里 clone() 是避免前向 ODE 解出来的 hidden states 受到外部影响,
        # 因为它们后续要被用作反向 ODE 的初值
        # ps: clone() 返回的 tensor 与原 tensor 不共享内存，也是就两者的修改互不影响,
        # clone 出来的 tensor 本身不会存储梯度, 因其在计算图中不是叶子节点而是中间节点, 但其梯度会叠加到原 tensor 上
        ctx.save_for_backward(t, z.clone(), flat_parameters)

        return z

    @staticmethod
    def backward(ctx, dLdz, ODEsolver: Callable = ODEsolver_Euler):
        """
        Custom backward pass, for calculating gradients with continous adjoint method.
        NOTE: In forward function, we output intermediate hidden states for loss computation,
        thus here will receive multiple gradients for each hidden state that `dLdz` shape: (time_len, batch_size, *z_shape)
        """

        func = ctx.func                            # f = dz/dt
        t, z, flat_parameters = ctx.saved_tensors  # (time_len,); (time_len, bs, *z_shape); (n_params,)

        time_len, batch_size, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Flatten dL/dz here for convenience. (将最后一维展平 *z_shape -> n_dim)
        dLdz = dLdz.view(time_len, batch_size, n_dim)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            建模 augmented 动态系统的微分方程
            tensors here are temporal slices
            t_i - is tensor with size: (1,)
            aug_z_i - is tensor with size: (bs, n_dim + n_dim + n_params + 1)
            """

            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2 * n_dim]  # z_{i}, a(z_{i})
            # Unflatten z and a
            z_i = z_i.view(batch_size, *z_shape)
            a = a.view(batch_size, *z_shape)

            with torch.set_grad_enabled(True):
                # 将 tensor 从计算图中剥离出来(`deteach()`)并设置为接受梯度,
                # 因为这里过来的 t_i 和 z_i 可能并非是计算图的叶子节点(因而没有梯度), 比如 z_i 就是由前向过程中 clone() 而来的
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)

                # Outputs evaluated result and vJps
                # 通过让神经网络做一次前向过程得到: f_{i} = dz_{i}/dt_{i};
                # 然后, 通过 torch.autograd 自动微分框架计算3个 vJps:
                # a(z_{i}) * df_{i}/dz_{i}, a(z_{i}) * df_{i}/dt_{i}, a(z_{i}) * df_{i}/dp
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)

                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(batch_size, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(batch_size, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(batch_size, 1).to(z_i)

            # 展平最后一维 *z_shape -> n_dim
            # 主要是外面的 z 和 a(z) 都做了这样的操作, 于是这里就是为了维度能够对应起来
            func_eval = func_eval.view(batch_size, n_dim)
            adfdz = adfdz.view(batch_size, n_dim)

            # NOTE: da(z)/dt = -a(z) * df/dz, da(p)/dt = -a(z) * df/dp, da(t)/dt = -a(z) * df/dt
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        # NOTE: 这整个大过程不需要梯度, 仅在计算 vJps 的时候 enable 梯度
        with torch.no_grad():
            # 为 hidden states, 网络参数 以及 时间变量 创建对应的伴随状态
            # NOTE: `adj_p` & `adj_t` 多了 `bs` 这一维度(按理说梯度应与张量的 shape 一致),
            # 为的是能够和 `adj_z` 拼起来作为 augmented states.
            # 多出来的这一维不要紧，最终会在这个维度上 sum 起来对应回张量的维度

            adj_z = torch.zeros(batch_size, n_dim).to(dLdz)     # a(z)
            adj_p = torch.zeros(batch_size, n_params).to(dLdz)  # a(p)
            # In contrast to z and p, we need to return gradients for all discreted times a(t0), a(t1), ..., a(T)
            adj_t = torch.zeros(time_len, batch_size, 1).to(dLdz)

            # From discreted step T to step 1(step 1 will result to step 0). [依次求解每个离散区间的反向 ODE]
            for i_t in range(time_len - 1, 0, -1):

                z_i = z[i_t]                                  # z_{i}
                t_i = t[i_t]                                  # t_{i}
                f_i = func(z_i, t_i).view(batch_size, n_dim)  # dz_{i}/dt_{i} (bs, *z_shape) -> (bs, n_dim)

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
                # (batch_size, n_dim + n_dim + n_params + 1)
                aug_z = torch.cat((z_i.view(batch_size, n_dim), adj_z, torch.zeros(batch_size, n_params).to(z), adj_t[i_t]), dim=-1)
                # 将多个状态(`aug_z`)和对应的微分方程组合在一起(`augmented_dynamics`)后，就可以求解器同时解多个反向 ODE
                # Solve augmented system backwards, get z_{i - 1}, a(z_{i - 1}), a(p_{i - 1}), a(t_{i - 1})
                aug_ans = ODEsolver(aug_z, t_i, t[i_t - 1], augmented_dynamics)

                # Unpack solved backwards augmented system (从解得的 augmented 状态中分离出各个伴随状态)
                adj_z[:] = aug_ans[:, n_dim:2 * n_dim]
                # Note: we should accumulated sum all discreted adjoints for params
                adj_p[:] += aug_ans[:, 2 * n_dim:(2 * n_dim + n_params)]
                adj_t[i_t - 1] = aug_ans[:, (2 * n_dim + n_params):]

                del aug_z, aug_ans

            # 在 t0 时刻也有 loss 过来的直接梯度，于是最后这里也要调整一下
            dLdz_0 = dLdz[0]                                                                          # dL/dz_{0}
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]  # dL/dt_{0}

            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0

        # 输出的数量需要和前向过程中输入的数量对应起来
        # Gradients for each input of forward.
        # dL/dz, dL/dt for all discrete times, dL/dp, None for function call.
        return adj_z.view(batch_size, *z_shape), adj_t, adj_p, None

class NeuralODE(nn.Module):
    ''' 回归到 torch.nn.Module，将 Neural ODE 的所有逻辑都封装在其中 '''
    def __init__(self, func: Union[nn.Module, Callable], ODEsolver: Callable = ODEsolver_Euler):
        super(NeuralODE, self).__init__()
        self.func = func  # 参数化的微分方程，在这里就是神经网络
        self.ODEsolver = ODEsolver  # ODE 求解器，默认是欧拉法

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence: bool = False):

        t = t.to(z0)  # 将 t 的 dtype & device 都变为与 z0 的一致
        # ODEAdjoint 是 `torch.autograd.Function` 而非 `torch.nn.Module`, 这是为了能够实现自定义的反向传播
        z = ODE_adjoint.apply(z0, t, self.func.flatten_parameters(), self.func, self.ODEsolver)

        return z if return_whole_sequence else z[-1]  # `return_whole_sequence` 代表是否返回所有离散时刻的输出状态
    
if __name__ == '__main__':
    # For debugging
    pass