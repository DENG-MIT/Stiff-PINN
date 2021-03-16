import torch
from torch import nn
from config import device, default_tensor_type

torch.set_default_tensor_type(default_tensor_type)


class LearnedTanh(nn.Module):
    def __init__(self, slope=1, n=1):
        super(LearnedTanh, self).__init__()
        self.slope = nn.Parameter(torch.Tensor(1).fill_(slope).to(device))
        self.n = torch.ones(1).to(device) * n

    def forward(self, input):
        return torch.tanh(self.n * self.slope * input)


class PINN_Model(nn.Module):
    def __init__(self, nodes=40, layers=2, y0=0, w_scale=None, x_scale=1):
        super(PINN_Model, self).__init__()

        self.y0 = y0
        self.w_scale = w_scale
        self.x_scale = x_scale

        self.activation = LearnedTanh(slope=0.1, n=10)

        self.seq = nn.Sequential()
        self.seq.add_module('fc_1', nn.Linear(1, nodes))
        self.seq.add_module('relu_1', self.activation)
        for i in range(layers):
            self.seq.add_module('fc_' + str(i + 2), nn.Linear(nodes, nodes))
            self.seq.add_module('relu_' + str(i + 2), self.activation)
        self.seq.add_module('fc_last', nn.Linear(nodes, self.y0.shape[1]))
        # self.seq.add_module('relu_last', nn.Softplus())

    def get_slope(self):

        return self.activation.slope.item() * self.activation.n.item()

    def forward(self, x):

        x_rescale = x / self.x_scale

        return self.y0 + self.w_scale * x_rescale * self.seq(torch.log(x_rescale))
