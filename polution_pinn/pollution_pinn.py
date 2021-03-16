# -*- coding: utf-8 -*-

import numpy as np
import torch
from tqdm import tqdm

from _utils import grad_norm, plot_pinn_y, to_np, K, ODE, get_solution, Get_Derivative
from config import cuda_index, device, default_tensor_type
from pinn_model import PINN_Model

import matplotlib.pyplot as plt
import math

torch.set_default_tensor_type(default_tensor_type)
# np.random.seed(2)
# torch.manual_seed(2)

# compt config
is_restart = False
use_softmax = False

n_grid_train = 500
n_grid_test = 100
learning_rate = 1e-3

num_epochs = 10
printout_freq = 1000

# Solving Robertson
ode = ODE(20)

y_list = []
dydt_list = []
np.random.seed(0)

order_start = -4
order_stop = math.log10(60)
delta_order = order_start - order_stop 

t_start = pow(10,order_start)
t_end = pow(10,order_stop)
n_steps = n_grid_train

t_np = np.logspace(start=order_start, stop=order_stop, num=n_steps, endpoint=True)
n_steps = t_np.shape[0]

#y0 = np.random.uniform(np.array([0.5, 0]), np.array([1.5, 0]))
y0 = np.zeros(20)
y0[1]  = 0.2
y0[3]  = 0.04
y0[6]  = 0.1
y0[7]  = 0.3
y0[8]  = 0.01
y0[16] = 0.007
print(y0)
y = get_solution(ode, y0, t_end, n_steps, t_np)

y_list.append(y[1:, :])
y_np = np.vstack(y_list)

fig = plt.figure(figsize=(9, 8))
for i in range(5):
    ax = fig.add_subplot(3, 3, i+1)
    ax.plot(t_np, y[1:,i], label='y{}'.format(i+1))
    ax.set_ylabel('State')
    ax.set_xscale('log')
    ax.set_xlim(1e-4, 60)
    ax.legend()
    ax.set_xlabel('Time')

fig.tight_layout()
#plt.savefig('./figs/y_CVode')
plt.show()


# Training PINN
t_true = torch.from_numpy(t_np)
t_true.to(device=device)

y_true = torch.from_numpy(y_np)
y_true.to(device=device)

n_var = 20

# initial condition
y0 = y_true[0,:].view(-1,n_var).to(device=device)

# Scaling factor
y_scale = y_true.max(dim=0).values.to(device=device)
x_scale = torch.Tensor([t_end]).to(device=device)
w_scale = torch.ones(n_var).to(device=device) * y_scale

checkpoint_path = 'models/robertson_stiff.pt'

net = PINN_Model(nodes=60, layers=5, y0=y0, w_scale=w_scale,
                x_scale=x_scale, use_softmax=use_softmax).to(device=device)


criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                            weight_decay=5e-4)

loss_list = {}
key_list_loss = ['res_train', 'res_test','grad_norm', 'slope']
key_list_alpha = []   
    
for key in key_list_loss + key_list_alpha:
    loss_list[key] = []

epoch_old = 0

# prepare data
t_end = t_true.max().item()
eps = 1e-30
eps_tensor = torch.Tensor([eps]).to(device=device)

# sampling equally in linear-scale
#x_train = (torch.rand(n_grid_train, device=device).unsqueeze(-1)) * t_end + eps
#x_test = (torch.rand(n_grid_test, device=device).unsqueeze(-1)) * t_end + eps

# sampling equally in log-scale
x_train = torch.pow(10, (torch.rand(n_grid_train, device=device).unsqueeze(-1) * delta_order)) * t_end + eps 
x_test = torch.pow(10, (torch.rand(n_grid_test, device=device).unsqueeze(-1) * delta_order)) * t_end + eps 

if is_restart is True:
    checkpoint = torch.load(checkpoint_path + '.tar', map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_old = checkpoint['epoch']
    loss_list = checkpoint['loss_list']
    x_train = torch.Tensor(checkpoint['x_train']).to(device)
    x_test = torch.Tensor(checkpoint['x_test']).to(device)



x_all = torch.cat([x_train, x_test], dim=0)
x_all_repeat = x_all.repeat(n_var, 1)
x_all_repeat.requires_grad = True

n_total = n_grid_train + n_grid_test

y_jac_indice = torch.empty([n_total * n_var, 2]).long()
for i in range(n_var):
    y_jac_indice[n_total * i:n_total * (i + 1), 0] = torch.arange(0, n_total) + n_total * i
    y_jac_indice[n_total * i:n_total * (i + 1), 1] = i

loss_res_train = torch.Tensor(n_var + 1).to(device=device)

rhs = Get_Derivative(n_grid_train+n_grid_test, n_var)

for epoch in tqdm(range(num_epochs)):
    if is_restart:
        if epoch < epoch_old:
            continue
    
    y_all_repeat = net(x_all_repeat).abs()

    # use jacobian tricks
    dydt_all = torch.autograd.grad(outputs=y_all_repeat[y_jac_indice[:, 0],
                                                        y_jac_indice[:, 1]].sum(),
                                  inputs=x_all_repeat,
                                  retain_graph=True,
                                  create_graph=True,
                                  allow_unused=True)[0].view(n_var, -1).T

    y_all = y_all_repeat[:n_total]
    
    
    
    shape = y_all.shape
    rhs_all = torch.Tensor(shape).to(device=device)
    rhs_all = rhs(y_all)
    #rhs_all = y_all
              
    y_train = y_all[:n_grid_train, :]
    y_test = y_all[n_grid_train:, :]
    rhs_train = rhs_all[:n_grid_train, :]
    rhs_test = rhs_all[n_grid_train:, :]
    dydt_train = dydt_all[:n_grid_train, :]
    dydt_test = dydt_all[n_grid_train:, :]

    loss_res_train = criterion(dydt_train, rhs_train)                              
    loss_res_test = criterion(dydt_test, rhs_test) 
    
    optimizer.zero_grad()  
    loss_res_train.backward(retain_graph=True)
    optimizer.step()

    # grad_sum = grad_norm(net)
    # slope = net.get_slope()

    # loss_list['res_train'].append(loss_train.item())
    # loss_list['res_test'].append(loss_test.item())
    # loss_list['slope'].append(slope)
    # loss_list['grad_norm'].append(grad_sum)

    # if epoch % printout_freq == 0:
    #     print('\n @epoch {} cuda {} slope {:.2f} grad_norm {:.2e}'.format(
    #         epoch, cuda_index, slope, grad_sum))
    #     print(['alpha {} = {:.2e}'.format(key, loss_list[key][epoch]) for key in key_list_alpha])
    #     print(['{} = {:.2e}'.format(key, loss_list[key][epoch]) for key in key_list_loss])
        
    #     # for plotting
    #     sorted_index = x_train.sort(dim=0).indices.view(-1)
        
    #     # plot here
    #     plot_pinn_y(to_np(x_train[sorted_index]),
    #                 to_np(y_train[sorted_index]),
    #                 to_np(t_true),
    #                 to_np(y_true),
    #                 to_np(dydt_train[sorted_index]),
    #                 to_np(rhs_train[sorted_index]),
    #                 loss_list,
    #               x_scale, t_start)

    #     torch.save({'epoch': epoch,
    #                 'model_state_dict': net.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'loss_list': loss_list,
    #                 'x_train': to_np(x_train),
    #                 'x_test': to_np(x_test),
    #                 }, checkpoint_path + '.tar')

        #torch.save(net, checkpoint_path)



