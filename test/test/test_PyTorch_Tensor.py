import torch

# 每一个Tensor都有 torch.dtype, torch.device, torch.layout 三个属性 (dtype: 数据类型; device: 计算设备; layout: 张量的存储方式)
# torch.device 标识了 torch.Tensor 对象在创建之后所存储在的设备名称
# torch.layout 表示了 torch.Tensor 内存布局的对象


if __name__ == '__main__':
    dim1, dim2 = 2,3

    # a = torch.Tensor([[1,2],[3,4]])
    a = torch.Tensor(dim1, dim2)  # 定义指定形状的张量 (初始值都是随机化生产的)
    # a = torch.ones(dim1,dim2)  # 定义全1张量
    # a = torch.zeros(dim1,dim2)  # 定义全0张量

    # b = torch.ones_like(a)
    b = torch.zeros_like(a)

    mean, std = 0,1
    # c = torch.rand(dim1,dim2)
    # c = torch.normal(mean=mean, std=std, size=(dim1,dim2))  # 定义均值为0，标准差为1的正态分布
    c = torch.Tensor(dim1, dim2).uniform_(-1,1)  # 定义均匀分布的张量 (先生成张量，再随机采样)

    ''' 序列 '''
    # d = torch.arange(0, 10, 3)  # 定义一个一维张量
    d = torch.linspace(2,10,4)  # 等差数列

    ''' 乱序 '''
    e = torch.randperm(10)  # 生成一个随机排列的张量

    ''' 稀疏张量 '''
    # coo 类型表示了非零元素的坐标形式
    ndim_sparse_1, ndim_sparse_2 = 100, 200
    indices = torch.tensor([[0, 1, 2], [2, 0, 1]])  # 稀疏张量的索引
    values = torch.tensor([3, 4, 5], dtype=torch.float32)  # 稀疏张量的值
    f = torch.sparse_coo_tensor(indices, values, (100, 200))  # 稀疏张量的形状
    print(f.shape)

    print(a)
    print(a.shape)   # 查看张量的形状
    print(a.type())  # 查看张量的类型

    print(b)
    print(b.shape)   # 查看张量的形状
    print(b.type())  # 查看张量的类型

    print(c)
    print(c.shape)   # 查看张量的形状
    print(c.type())  # 查看张量的类型

    print(d)
    print(d.shape)   # 查看张量的形状
    print(d.type())  # 查看张量的类型

    print(e)
    print(e.shape)   # 查看张量的形状
    print(e.type())  # 查看张量的类型