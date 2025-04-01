# This code contains basic functions for data processing and visualization.

import torch

# 此函数可以将torch.Tensor转换为numpy.ndarray
def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x