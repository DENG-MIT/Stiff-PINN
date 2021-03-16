# -*- coding: utf-8 -*-

import numpy as np
import torch
from tqdm import tqdm

from _utils import grad_norm, plot_pinn_y, to_np, K, ODE, get_solution, get_param_grad, grad_mean
from config import cuda_index, device, default_tensor_type
from pinn_model import PINN_Model

torch.set_default_tensor_type(default_tensor_type)
# np.random.seed(2)
# torch.manual_seed(2)

is_restart = False
use_annealing = True
n_grid_train = 2500
n_grid_test = 500
learning_rate = 1e-3

num_epochs = 100000
printout_freq = 1000

update_lambda_freq = 100
alpha = 0.7

# Solving Robertson
ode = ODE(3)

y_list = []
dydt_list = []
np.random.seed(0)

t_end = 1e-2
n_steps = n_grid_train

t_np = np.logspace(start=-8, stop=-2, num=n_steps, endpoint=True)
n_steps = t_np.shape[0]

y0 = np.random.uniform(np.array([0.5, 0, 0]), np.array([1.5, 0, 0]))
print(y0)
y = get_solution(ode, y0, t_end, n_steps, t_np)
y_list.append(y[1:, :])

y_np = np.vstack(y_list)

np.savez('./Datasets/Robertson_PINN Dataset.npz', t=t_np, y=y_np)


# Training PINN
t_true = torch.from_numpy(t_np)
t_true.to(device=device)

y_true = torch.from_numpy(y_np)
y_true.to(device=device)

n_var = 3

# initial condition
y0 = y_true[0,:].view(-1,n_var).to(device=device)

# Scaling factor
y_scale = y_true.max(dim=0).values.to(device=device)
#x_scale = t_true.max(dim=0).values.to(device=device)
x_scale = torch.Tensor([1]).to(device=device)
w_res = torch.ones(n_var).to(device=device) * x_scale / y_scale
w_scale = torch.ones(n_var).to(device=device) * y_scale


checkpoint_path = 'models/robertson_stiff.pt'

net = PINN_Model(nodes=15, layers=4, y0=y0, w_scale=w_scale,
                 x_scale=x_scale).to(device=device)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                             weight_decay=5e-4)

loss_list = {}
key_list_loss = ['res_train', 'res_test', 
                 'grad_norm', 'slope','res_0','res_1','res_2',
                 'lambda0','lambda1','lambda2']
key_list_alpha = []
for key in key_list_loss + key_list_alpha:
    loss_list[key] = []

epoch_old = 0

# prepare data
t_end = t_true.max().item()
eps = 1e-30

# sampling equally in linear-scale
x_train = (torch.rand(n_grid_train, device=device).unsqueeze(-1)) * t_end + eps
x_test = (torch.rand(n_grid_test, device=device).unsqueeze(-1)) * t_end + eps

# sampling equally in log-scale
x_train = torch.pow(10, (torch.rand(n_grid_train, device=device).unsqueeze(-1) * (-6))) * t_end + eps # noqa: E501
x_test = torch.pow(10, (torch.rand(n_grid_test, device=device).unsqueeze(-1) * (-6))) * t_end + eps # noqa: E501


if is_restart is True:
    checkpoint = torch.load(checkpoint_path + '.tar', map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_old = checkpoint['epoch']
    loss_list = checkpoint['loss_list']
    x_train = torch.Tensor(checkpoint['x_train']).to(device)
    x_test = torch.Tensor(checkpoint['x_test']).to(device)

# for plotting
sorted_index = x_train.sort(dim=0).indices.view(-1)

x_all = torch.cat([x_train, x_test], dim=0)
x_all_repeat = x_all.repeat(n_var, 1)
x_all_repeat.requires_grad = True

n_total = n_grid_train + n_grid_test

y_jac_indice = torch.empty([n_total * n_var, 2]).long()
for i in range(n_var):
    y_jac_indice[n_total * i:n_total * (i + 1), 0] = torch.arange(0, n_total) + n_total * i
    y_jac_indice[n_total * i:n_total * (i + 1), 1] = i

loss_res_train = torch.Tensor(3).to(device=device)

if use_annealing == False:
    scaling_fac = 1e-7
    I1 = torch.Tensor([1,0,0]).to(device=device) * scaling_fac
    I2 = torch.Tensor([0,1e-9,0]).to(device=device) * scaling_fac
    I3 = torch.Tensor([0,0,1e5]).to(device=device) * scaling_fac

if use_annealing == True:
    
    I1 = torch.Tensor([1e-6,0,0]).to(device=device) 
    I2 = torch.Tensor([0,1e-6,0]).to(device=device) 
    I3 = torch.Tensor([0,0,1e-6]).to(device=device) 
    
    # the first element is always set to 1
    lambda_hat = torch.ones_like(loss_res_train).to(device=device) 
    
#    trainable_num = 87
#                
#    grad_tensor = torch.Tensor(trainable_num,3)

for epoch in tqdm(range(num_epochs)):
    if is_restart:
        if epoch < epoch_old:
            continue

    y_all_repeat = net(x_all_repeat)

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
    
    rhs_all[:,0] = -K[0] * y_all[:,0] + K[2] * y_all[:,1] * y_all[:,2]
    rhs_all[:,1] = K[0] * y_all[:,0] - K[2] * y_all[:,1] * y_all[:,2] \
                    - K[1] * y_all[:,1] * y_all[:,1]
    rhs_all[:,2] = K[1] * y_all[:,1] * y_all[:,1]

    y_train = y_all[:n_grid_train, :]
    y_test = y_all[n_grid_train:, :]
    rhs_train = rhs_all[:n_grid_train, :]
    rhs_test = rhs_all[n_grid_train:, :]
    dydt_train = dydt_all[:n_grid_train, :]
    dydt_test = dydt_all[n_grid_train:, :]

    if use_annealing == False:
        
        loss_res_train = I1 * criterion(dydt_train[:,0] * w_res[0], rhs_train[:,0] * w_res[0]) \
                    + I2 *  criterion(dydt_train[:,1] * w_res[1], \
                  rhs_train[:,1] * w_res[1]) + \
                    I3 * criterion(dydt_train[:,2] * w_res[2], \
                  rhs_train[:,2] * w_res[2])
                                     
        loss_res_test = I1 * criterion(dydt_test[:,0] * w_res[0], rhs_test[:,0] * w_res[0]) \
                    + I2 *  criterion(dydt_test[:,1] * w_res[1], \
                  rhs_test[:,1] * w_res[1]) + \
                    I3 * criterion(dydt_test[:,2] * w_res[2], \
                  rhs_test[:,2] * w_res[2])
        
        loss_train = loss_res_train.sum()
        loss_test = loss_res_test.sum()
        
    elif use_annealing == True:
        
        loss_res_train = I1 * criterion(dydt_train[:,0], rhs_train[:,0]) \
                        + I2 *  criterion(dydt_train[:,1], rhs_train[:,1] ) + \
                        I3 * criterion(dydt_train[:,2], rhs_train[:,2])
        
        if epoch % update_lambda_freq == 0:
                   
#            loss_res_train_annealing0 = loss_res_train[0]
#            optimizer.zero_grad() 
#            loss_res_train_annealing0.backward(retain_graph=True)
#            grad_tensor[:,0] = get_param_grad(net,trainable_num)
#            
#            loss_res_train_annealing1 = loss_res_train[1]
#            optimizer.zero_grad() 
#            loss_res_train_annealing1.backward(retain_graph=True)
#            grad_tensor[:,1] = get_param_grad(net,trainable_num)
#            
#            loss_res_train_annealing2 = loss_res_train[2]
#            optimizer.zero_grad() 
#            loss_res_train_annealing2.backward(retain_graph=True)
#            grad_tensor[:,2] = get_param_grad(net,trainable_num)
#            
#            max_grad0 = torch.max(grad_tensor[:,0])
#            lambda_hat[1] = (1 - alpha) * lambda_hat[1] + \
#                            alpha * max_grad0 / torch.mean(grad_tensor[:,1])
#            lambda_hat[2] = (1 - alpha) * lambda_hat[2] + \
#                            alpha * max_grad0 / torch.mean(grad_tensor[:,2])
            
            loss_res_train_annealing0 = loss_res_train[0]
            optimizer.zero_grad() 
            loss_res_train_annealing0.backward(retain_graph=True)
            grad_mean0 = grad_mean(net)
            
            loss_res_train_annealing1 = loss_res_train[1]
            optimizer.zero_grad() 
            loss_res_train_annealing1.backward(retain_graph=True)
            grad_mean1 = grad_mean(net)
            
            loss_res_train_annealing2 = loss_res_train[2]
            optimizer.zero_grad() 
            loss_res_train_annealing2.backward(retain_graph=True)
            grad_mean2 = grad_mean(net)
            
            lambda_hat[1] = (1 - alpha) * lambda_hat[1] + \
                            alpha * grad_mean0 / grad_mean1
            lambda_hat[2] = (1 - alpha) * lambda_hat[2] + \
                            alpha * grad_mean0 / grad_mean2
        
        loss_res_train = loss_res_train * lambda_hat
        
        loss_res_test = (I1 * criterion(dydt_test[:,0], rhs_test[:,0]) \
                    + I2 *  criterion(dydt_test[:,1], rhs_test[:,1]) + \
                    I3 * criterion(dydt_test[:,2], rhs_test[:,2])) * lambda_hat
        
        loss_train = loss_res_train.sum()
        loss_test = loss_res_test.sum()
        
        loss_list['lambda0'].append(lambda_hat[0].item())
        loss_list['lambda1'].append(lambda_hat[1].item())
        loss_list['lambda2'].append(lambda_hat[2].item())

    optimizer.zero_grad()  
    loss_train.backward()
    optimizer.step()

    grad_sum = grad_norm(net)
    slope = net.get_slope()

    loss_list['res_train'].append(loss_train.item())
    loss_list['res_test'].append(loss_test.item())
    loss_list['res_0'].append(loss_res_train[0].item())
    loss_list['res_1'].append(loss_res_train[1].item())
    loss_list['res_2'].append(loss_res_train[2].item())
    loss_list['slope'].append(slope)
    loss_list['grad_norm'].append(grad_sum)

    if epoch % printout_freq == 0:
        print('\n @epoch {} cuda {} slope {:.2f} grad_norm {:.2e}'.format(
            epoch, cuda_index, slope, grad_sum))
        print(['alpha {} = {:.2e}'.format(key, loss_list[key][epoch]) for key in key_list_alpha])
        print(['{} = {:.2e}'.format(key, loss_list[key][epoch]) for key in key_list_loss])

        # plot here
        plot_pinn_y(to_np(x_train[sorted_index]),
                    to_np(y_train[sorted_index]),
                    to_np(t_true),
                    to_np(y_true),
                    to_np(dydt_train[sorted_index]),
                    to_np(rhs_train[sorted_index]),
                    loss_list,
                    x_scale)

        torch.save({'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_list': loss_list,
                    'x_train': to_np(x_train),
                    'x_test': to_np(x_test),
                    }, checkpoint_path + '.tar')

        torch.save(net, checkpoint_path)


