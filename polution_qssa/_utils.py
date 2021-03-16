import cantera as ct
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import torch

import numpy as np
from assimulo.solvers import RodasODE
from assimulo.problem import Explicit_Problem

from config import device

#matplotlib.use('TkAgg')


torch.set_default_tensor_type("torch.DoubleTensor")

K = [0.35, 0.266e2, 0.123e5, 0.86e-3, 0.82e-3,\
     0.15e5, 0.13e-3, 0.24e5, 0.165e5, 0.9e4,\
     0.22e-1, 0.12e5, 0.188e1, 0.163e5, 0.48e7,\
     0.35e-3, 0.175e-1, 0.1e9, 0.444e12, 0.124e4,\
     0.210e1, 0.578e1, 0.474e-1, 0.178e4, 0.312e1
     ]


class ODE():
    def __init__(self, species_count):
        self.species_count = species_count
        self.dx_dt = np.zeros(species_count)
        self.r = np.zeros(25)
        
    def __call__(self, t, x):
        
        self.r[0] = K[0] * x[0]
        self.r[1] = K[1] * x[1] * x[3]
        self.r[2] = K[2] * x[4] * x[1]
        self.r[3] = K[3] * x[6]
        self.r[4] = K[4] * x[6]
        self.r[5] = K[5] * x[5] * x[6]
        self.r[6] = K[6] * x[8]
        self.r[7] = K[7] * x[5] * x[8]
        self.r[8] = K[8] * x[1] * x[10]
        self.r[9] = K[9] * x[0] * x[10]
        self.r[10] = K[10] * x[12]
        self.r[11] = K[11] * x[1] * x[9]
        self.r[12] = K[12] * x[13]
        self.r[13] = K[13] * x[0] * x[5]
        self.r[14] = K[14] * x[2]
        self.r[15] = K[15] * x[3]
        self.r[16] = K[16] * x[3]
        self.r[17] = K[17] * x[15] 
        self.r[18] = K[18] * x[15]
        self.r[19] = K[19] * x[5] * x[16]
        self.r[20] = K[20] * x[18]
        self.r[21] = K[21] * x[18] 
        self.r[22] = K[22] * x[0] * x[3]
        self.r[23] = K[23] * x[0] * x[18]
        self.r[24] = K[24] * x[19]
          
        self.dx_dt[0] = -self.r[0] - self.r[9] - self.r[13] - self.r[22] - self.r[23] \
                        + self.r[1] + self.r[2] + self.r[8] + self.r[10] + self.r[11] \
                        + self.r[21] + self.r[24]
        self.dx_dt[1] = - self.r[1] - self.r[2] - self.r[8] - self.r[11] \
                        + self.r[0] + self.r[20]
        self.dx_dt[2] = - self.r[14] + self.r[0] + self.r[16] + self.r[18] \
                        + self.r[21] 
        self.dx_dt[3] = - self.r[1] - self.r[15] - self.r[16] - self.r[22] \
                        + self.r[14] 
        self.dx_dt[4] = - self.r[2] + 2 * self.r[3] + self.r[5] + self.r[6] \
                        + self.r[12] + self.r[19]
        self.dx_dt[5] = - self.r[5] - self.r[7] - self.r[13] - self.r[19] \
                        + self.r[2] + 2 * self.r[17]
        self.dx_dt[6] = - self.r[3] - self.r[4] - self.r[5] + self.r[12] 
        self.dx_dt[7] = self.r[3] + self.r[4] + self.r[5] + self.r[6] 
        self.dx_dt[8] = - self.r[6] - self.r[7] 
        self.dx_dt[9] = - self.r[11] + self.r[6] + self.r[8] 
        self.dx_dt[10] = - self.r[8] - self.r[9] + self.r[7] + self.r[10] 
        self.dx_dt[11] = self.r[8] 
        self.dx_dt[12] = - self.r[10] + self.r[9] 
        self.dx_dt[13] = - self.r[12] + self.r[11] 
        self.dx_dt[14] = self.r[13] 
        self.dx_dt[15] = - self.r[17] - self.r[18] + self.r[15] 
        self.dx_dt[16] = - self.r[19]
        self.dx_dt[17] = self.r[19] 
        self.dx_dt[18] = - self.r[20] - self.r[21] - self.r[23] + self.r[22] \
                        + self.r[24]
        self.dx_dt[19] = - self.r[24] + self.r[23] 

        return self.dx_dt


def get_solution(ode, y0, t_end, n_steps, t_np):
    '''Use solve_ivp from scipy to solve the ODE'''

    exp_mod = Explicit_Problem(
        ode, y0, name='Robertson Chemical Kinetics Example')

    # Create an Assimulo explicit solver (Euler)
    #exp_sim = ExplicitEuler(exp_mod)
    exp_sim = RodasODE(exp_mod)

    # Sets the solver paramters
#    exp_sim.iter = 'Newton'
#    exp_sim.discr = 'BDF'
    # exp_sim.rtol = 1.e-10
    # exp_sim.atol = 1.e-10
    # exp_sim.inith = 1.e-6
    
    exp_sim.usejac = False

    # Simulate
    # ncp = 0 will print the internal time step
    t, y = exp_sim.simulate(tfinal=t_end,ncp=0, ncp_list=t_np)

    return y

class Get_Derivative():
    ''' calc the rhs derivatives for PINN training
    '''
    def __init__(self,  n_points, species_count):
        self.species_count = species_count
        self.dx_dt = torch.zeros(n_points, species_count)
        self.dx_dt_temp = torch.zeros(n_points, species_count)
        self.r = torch.zeros(n_points, 25)
        diag0 = torch.ones(25)
        self.I0 = torch.diag(diag0)
        diag1 = torch.ones(species_count)
        self.I1 = torch.diag(diag1)
    
    def __call__(self, x):
        
        self.r =  self.I0[0,:] * K[0] * x[:,0:1] \
                + self.I0[1,:] * K[1] * x[:,1:2] * x[:,3:4] \
                + self.I0[2,:] * K[2] * x[:,4:5] * x[:,1:2] \
                + self.I0[3,:] * K[3] * x[:,6:7] \
                + self.I0[4,:] * K[4] * x[:,6:7] \
                + self.I0[5,:] * K[5] * x[:,5:6] * x[:,6:7] \
                + self.I0[6,:] * K[6] * x[:,8:9] \
                + self.I0[7,:] * K[7] * x[:,5:6] * x[:,8:9] \
                + self.I0[8,:] * K[8] * x[:,1:2] * x[:,10:11] \
                + self.I0[9,:] * K[9] * x[:,0:1] * x[:,10:11] \
                + self.I0[10,:] * K[10] * x[:,12:13] \
                + self.I0[11,:] * K[11] * x[:,1:2] * x[:,9:10] \
                + self.I0[12,:] * K[12] * x[:,13:14] \
                + self.I0[13,:] * K[13] * x[:,0:1] * x[:,5:6] \
                + self.I0[14,:] * K[14] * x[:,2:3] \
                + self.I0[15,:] * K[15] * x[:,3:4] \
                + self.I0[16,:] * K[16] * x[:,3:4] \
                + self.I0[17,:] * K[17] * x[:,15:16] \
                + self.I0[18,:] * K[18] * x[:,15:16] \
                + self.I0[19,:] * K[19] * x[:,5:6] * x[:,16:17] \
                + self.I0[20,:] * K[20] * x[:,18:19] \
                + self.I0[21,:] * K[21] * x[:,18:19] \
                + self.I0[22,:] * K[22] * x[:,0:1] * x[:,3:4] \
                + self.I0[23,:] * K[23] * x[:,0:1] * x[:,18:19] \
                + self.I0[24,:] * K[24] * x[:,19:20]      
        
        self.dx_dt = (-self.r[:,0] - self.r[:,9] - self.r[:,13] - self.r[:,22] - self.r[:,23] \
                        + self.r[:,1] + self.r[:,2] + self.r[:,8] + self.r[:,10] + self.r[:,11] \
                        + self.r[:,21] + self.r[:,24]).view(-1,1) * self.I1[0,:] \
                        + (-self.r[:,1] - self.r[:,2] - self.r[:,8] - self.r[:,11] \
                        + self.r[:,0] + self.r[:,20]).view(-1,1) * self.I1[1,:] \
                        + (-self.r[:,14] + self.r[:,0] + self.r[:,16] + self.r[:,18] \
                        + self.r[:,21]).view(-1,1) * self.I1[2,:] \
                        + (-self.r[:,1] - self.r[:,15] - self.r[:,16] - self.r[:,22] \
                        + self.r[:,14]).view(-1,1) * self.I1[3,:] \
                        + (-self.r[:,2] + 2 * self.r[:,3] + self.r[:,5] + self.r[:,6] \
                        + self.r[:,12] + self.r[:,19]).view(-1,1) * self.I1[4,:] \
                        + (-self.r[:,5] - self.r[:,7] - self.r[:,13] - self.r[:,19] \
                        + self.r[:,2] + 2 * self.r[:,17]).view(-1,1) * self.I1[5,:] \
                        + (- self.r[:,3] - self.r[:,4] - self.r[:,5] + self.r[:,12]).view(-1,1) * self.I1[6,:] \
                        + (self.r[:,3] + self.r[:,4] + self.r[:,5] + self.r[:,6]).view(-1,1) * self.I1[7,:] \
                        + (-self.r[:,6] - self.r[:,7]).view(-1,1) * self.I1[8,:] \
                        + (-self.r[:,11] + self.r[:,6] + self.r[:,8]).view(-1,1) * self.I1[9,:] \
                        + (-self.r[:,8] - self.r[:,9] + self.r[:,7] + self.r[:,10]).view(-1,1) * self.I1[10,:] \
                        + self.r[:,8].view(-1,1) * self.I1[11,:] \
                        + (-self.r[:,10] + self.r[:,9]).view(-1,1) * self.I1[12,:] \
                        + (-self.r[:,12] + self.r[:,11]).view(-1,1) * self.I1[13,:] \
                        + self.r[:,13].view(-1,1) * self.I1[14,:] \
                        + (-self.r[:,17] - self.r[:,18] + self.r[:,15]).view(-1,1) * self.I1[15,:] \
                        + (-self.r[:,19]).view(-1,1) * self.I1[16,:] \
                        + self.r[:,19].view(-1,1) * self.I1[17,:] \
                        + (-self.r[:,20] - self.r[:,21] - self.r[:,23] + self.r[:,22] \
                        + self.r[:,24]).view(-1,1) * self.I1[18,:] \
                        + (-self.r[:,24] + self.r[:,23]).view(-1,1) * self.I1[19,:] 

        return self.dx_dt

def get_param_grad(pinn_model, param_nums):
    
    grad_norm = torch.ones(param_nums)
    
    count = 0
    
    for p in pinn_model.parameters():
        
        if p.requires_grad:
            
            for q in p.grad.data:
                
                grad_norm[count] = torch.norm(q,p=2)
                
                count += 1
    
    return grad_norm
    

def to_np(y):
    return y.detach().cpu().numpy()

def grad_norm(pinn_model):

    total_norm = 0

    for p in pinn_model.parameters():

        param_norm = p.grad.data.norm(2)

        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** (1. / 2)

    return total_norm


def grad_mean(pinn_model):

    total_norm = 0

    count = 0

    for p in pinn_model.parameters():
        param_norm = p.grad.data.norm(1)
        total_norm += param_norm.item()
        count += torch.numel(p.grad.data)

    grad_mean = total_norm / count

    return grad_mean


def plot_pinn_y(x_pinn, y_pinn, t_true, y_true,
                dydt, rhs, loss_list, x_scale, xlim_left):

    species_names = np.array(['0','1','2'])


    fig = plt.figure(figsize=(9, 9))
    for i in range(2):
        i_species = i

        ax = fig.add_subplot(3, 3, i + 1)
        ax.plot(x_pinn / x_scale.cpu(),
                y_pinn[:, i_species],
                color="r",
                ls="-",
                label='pinn')

        ax.plot(t_true / x_scale.cpu(),
                y_true[:, i_species],
                color="b",
                ls="-",
                marker=None,
                fillstyle='none',
                markersize=6,
                alpha=1.0,
                label='true')

        ax.set_xlim(left=xlim_left)
        ax.set_xscale('log')

        ax2 = ax.twinx()
        ax2.plot(x_pinn / x_scale.cpu(),
                 dydt[:, i_species],
                 ls='--',
                 color='r',
                 alpha=0.8,
                 label='dydt')

        ax2.plot(x_pinn / x_scale.cpu(),
                 rhs[:, i_species],
                 ls='--',
                 color='b',
                 alpha=0.8,
                 label='rhs')

        ax.set_title(species_names[i])
        if i == 0:
            ax.legend(loc='upper left', shadow=False, framealpha=0.5)
        if i == 1:
            ax2.legend(loc='upper right', shadow=False, framealpha=0.5)

    # plot loss
    key_list_loss = ['res_train', 'res_test']

    ax = fig.add_subplot(3, 3, i + 2)
    epoch = len(loss_list['res_train'])
    for key in key_list_loss:
        ax.plot(np.arange(epoch) + 1, loss_list[key], '-', lw=2, label=key)
    ax.legend(loc='upper right', shadow=False, framealpha=0.3)
    ax.set_yscale("log")
    if epoch > 2000:
        ax.set_xscale("log")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    
#    
#    key_list_loss2 = ['res_0','res_1']
#
#    ax = fig.add_subplot(3, 3, i + 3)
#    epoch = len(loss_list['res_0'])
#    for key in key_list_loss2:
#        ax.plot(np.arange(epoch) + 1, loss_list[key], '-', lw=2, label=key)
#    ax.legend(loc='upper right', shadow=False, framealpha=0.3)
#    ax.set_yscale("log")
#    if epoch > 2000:
#        ax.set_xscale("log")
#    ax.set_xlabel('Epoch')
#    ax.set_ylabel('Loss')

    key_list_loss3 = ['grad_norm']

    ax = fig.add_subplot(3, 3, i + 4)
    epoch = len(loss_list['grad_norm'])
    for key in key_list_loss3:
        ax.plot(np.arange(epoch) + 1, loss_list[key], '-', lw=2, label=key)
    ax.legend(loc='upper right', shadow=False, framealpha=0.3)
    ax.set_yscale("log")
    if epoch > 2000:
        ax.set_xscale("log")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Grad norm')

    key_list_loss4 = ['slope']

    ax = fig.add_subplot(3, 3, i + 5)
    epoch = len(loss_list['slope'])
    for key in key_list_loss4:
        ax.plot(np.arange(epoch) + 1, loss_list[key], '-', lw=2, label=key)
    ax.legend(loc='upper right', shadow=False, framealpha=0.3)
    ax.set_yscale("log")
    if epoch > 2000:
        ax.set_xscale("log")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Slope')

    fig.tight_layout()

    plt.show()

    figname = 'figs/pinn_loss.png'
    fig.savefig(figname, dpi=120, facecolor='w', edgecolor='w', orientation='portrait')

    ax.cla()
    fig.clf()
    plt.close()
