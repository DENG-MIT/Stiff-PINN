import matplotlib.pyplot as plt
import numpy as np
import torch

from config import device
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem


torch.set_default_tensor_type("torch.DoubleTensor")

K = [0.04, 3.E7, 1.E4]

class ODE():
    def __init__(self, species_count):
        self.species_count = species_count
        self.dx_dt = np.zeros(species_count)

    def __call__(self, t, x):
        self.dx_dt[0] = -K[0] * x[0] + K[2] * x[1] * x[2]
        self.dx_dt[1] = K[0] * x[0] - K[1] * x[1] ** 2 - K[2] * x[1] * x[2]
        self.dx_dt[2] = K[1] * x[1] ** 2

        return self.dx_dt


def get_solution(ode, y0, t_end, n_steps, t_np):
    '''Use solve_ivp from scipy to solve the ODE'''

    exp_mod = Explicit_Problem(
        ode, y0, name='Robertson Chemical Kinetics Example')

    # Create an Assimulo explicit solver (CVode)
    exp_sim = CVode(exp_mod)

    # Sets the solver paramters
    exp_sim.iter = 'Newton'
    exp_sim.discr = 'BDF'
    exp_sim.rtol = 1.e-4
    exp_sim.atol = np.array([1.0e-8, 1.0e-14, 1.0e-6])

    # Simulate
    # ncp = 0 will print the internal time step
    t, y = exp_sim.simulate(tfinal=t_end, ncp=0, ncp_list=t_np)

    return y


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
                dydt, rhs, loss_list, x_scale):
    species_names = np.array(['0', '2'])

    fig = plt.figure(figsize=(9, 9))

    ax = fig.add_subplot(3, 3, 1)
    ax.plot(x_pinn / x_scale,
            y_pinn[:, 0],
            color="r",
            ls="-",
            label='pinn')
    ax.plot(t_true / x_scale,
            y_true[:, 0],
            color="b",
            ls="-",
            marker=None,
            fillstyle='none',
            markersize=6,
            alpha=1.0,
            label='true')
    ax.set_xlim(left=1e-2)
    ax.set_xscale('log')
    ax2 = ax.twinx()
    ax2.plot(x_pinn / x_scale,
             dydt[:, 0],
             ls='--',
             color='r',
             alpha=0.8,
             label='dydt')
    ax2.plot(x_pinn / x_scale,
             rhs[:, 0],
             ls='--',
             color='b',
             alpha=0.8,
             label='rhs')
    ax.set_title(species_names[0])
    ax.legend(loc='upper left', shadow=False, framealpha=0.5)


    ax = fig.add_subplot(3, 3, 2)
    ax.plot(x_pinn / x_scale,
            y_pinn[:, 1],
            color="r",
            ls="-",
            label='pinn')
    ax.plot(t_true / x_scale,
            y_true[:, 2],
            color="b",
            ls="-",
            marker=None,
            fillstyle='none',
            markersize=6,
            alpha=1.0,
            label='true')
    ax.set_xlim(left=1e-2)
    ax.set_xscale('log')
    ax2 = ax.twinx()
    ax2.plot(x_pinn / x_scale,
             dydt[:, 1],
             ls='--',
             color='r',
             alpha=0.8,
             label='dydt')
    ax2.plot(x_pinn / x_scale,
             rhs[:, 1],
             ls='--',
             color='b',
             alpha=0.8,
             label='rhs')
    ax.set_title(species_names[1])
    ax2.legend(loc='upper right', shadow=False, framealpha=0.5)

    '''
    ax = fig.add_subplot(3, 3, 3)
    ax.plot(x_pinn / x_scale,
            y_pinn[:, 2],
            color="r",
            ls="-",
            label='pinn')
    ax.plot(t_true / x_scale,
            y_true[:, 2],
            color="b",
            ls="-",
            marker=None,
            fillstyle='none',
            markersize=6,
            alpha=1.0,
            label='true')
    ax.set_xlim(left=1e-7)
    ax.set_xscale('log')
    ax2 = ax.twinx()
    ax2.plot(x_pinn / x_scale,
             dydt[:, 1],
             ls='--',
             color='r',
             alpha=0.8,
             label='dydt')
    ax2.plot(x_pinn / x_scale,
             rhs[:, 1],
             ls='--',
             color='b',
             alpha=0.8,
             label='rhs')
    ax.set_title(species_names[1])
    ax2.legend(loc='upper right', shadow=False, framealpha=0.5)'''

    i = 2
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

    key_list_loss2 = ['res_0', 'res_2']

    ax = fig.add_subplot(3, 3, i + 3)
    epoch = len(loss_list['res_train'])
    for key in key_list_loss2:
        ax.plot(np.arange(epoch) + 1, loss_list[key], '-', lw=2, label=key)
    ax.legend(loc='upper right', shadow=False, framealpha=0.3)
    ax.set_yscale("log")
    if epoch > 2000:
        ax.set_xscale("log")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

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

    fig.tight_layout()

    figname = 'figs/pinn_loss.png'
    fig.savefig(figname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait')

    ax.cla()
    fig.clf()
    plt.close()
