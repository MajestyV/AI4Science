import torch
from torch import Tensor

if __name__ == '__main__':
    ndim1, ndim2 = 2, 3

    a = torch.rand(ndim1, ndim2)  # 定义指定形状的张量 (初始值都是随机化生产的)
    b = torch.rand(ndim1, ndim2)  # 定义均匀分布的张量 (先生成张量，再随机采样)

    c = a * b  # 在pytorch中，* 代表了哈达玛积

    print(a)
    print(b)
    print(c)

    print('-------------------------------------- 分割线 -------------------------------------------')

    # 矩阵运算
    d = torch.rand(2,3)
    e = torch.rand(3,2)

    print(d)
    print(e)

    # 矩阵乘法
    print(torch.mm(d, e))
    print(torch.matmul(d, e))  # 矩阵乘法
    print(d @ e)
    print(d.matmul(e))
    print(d.mm(e))