import torch

cuda_index = 2

cpu = torch.device('cpu')     # Default CUDA device
cuda = torch.device('cuda:{}'.format(cuda_index))     # Default CUDA device
device = cuda

if torch.cuda.is_available() is False:
    device = cpu


default_tensor_type = "torch.DoubleTensor"
