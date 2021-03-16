import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from config import device

torch.set_default_tensor_type("torch.DoubleTensor")


class MyDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]


class LearnedTanh(nn.Module):
    def __init__(self, slope=1, n=1):
        super(LearnedTanh, self).__init__()
        self.slope = nn.Parameter(torch.Tensor(1).fill_(slope).to(device))
        self.n = torch.ones(1).to(device) * n

    def forward(self, input):
        #return torch.tanh(self.n * self.slope * input)
        return nn.functional.gelu(self.n * self.slope * input)


class PINN_Model(nn.Module):
    def __init__(self, nodes=40, layers=2, y0=0, w_scale=None, x_scale=1):
        super(PINN_Model, self).__init__()

        self.y0 = y0
        self.w_scale = w_scale
        self.x_scale = x_scale

        self.activation = LearnedTanh(slope=1e-8, n=10)
        self.seq = nn.Sequential()
        self.seq.add_module('fc_1', nn.Linear(1, nodes))
        self.seq.add_module('relu_1', self.activation)
        for i in range(layers):
            self.seq.add_module('fc_' + str(i + 2), nn.Linear(nodes, nodes))
            self.seq.add_module('relu_' + str(i + 2), self.activation)
        self.seq.add_module('fc_last', nn.Linear(nodes, self.y0.shape[1]))
        # self.seq.add_module('relu_last', nn.Softplus())

    def xavier_init(self):

        for m in self._modules['seq']:

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def constant_init(self, w0):

        for m in self._modules['seq']:

            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, w0)
                nn.init.constant_(m.bias, w0)

    def get_slope(self):

        return self.activation.slope.item() * self.activation.n.item()

    def forward(self, x):

        return self.seq(torch.log(x / self.x_scale)) * (x / self.x_scale) * self.w_scale + self.y0
        #return self.seq(x / self.x_scale) * (x / self.x_scale) * self.w_scale + self.y0
