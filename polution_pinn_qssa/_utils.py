import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import device

torch.set_default_tensor_type("torch.DoubleTensor")


def to_np(y):
    return y.detach().cpu().numpy()


def get_idt(gas, P, composition, T, energy='on'):

    gas.TPX = T, P, composition

    r = ct.IdealGasConstPressureReactor(gas, energy=energy)

    sim = ct.ReactorNet([r])

    idt = 0

    t_end = 1e-0

    states = ct.SolutionArray(gas, extra=["t"])

    while sim.time < t_end:
        sim.step()
        states.append(r.thermo.state, t=sim.time)

        # Ignition creterior: temperature or species
        if energy == 'on':

            if r.thermo.T > T + 600 and idt < 1e-10:
                idt = sim.time

        else:
            # index 0 is the species O2 (Oxygen)
            if r.thermo.Y[0] < 1e-3 and idt < 1e-10:
                idt = sim.time

        if idt > 1e-10 and sim.time > 2 * idt:
            break

    if idt < 1e-10:
        idt = t_end

    states_np = np.hstack((np.atleast_2d(states.T).T, states.Y))

    t_tensor = torch.Tensor(states.t).view(-1, 1).to(device=device)

    TY_tensor = torch.Tensor(states_np).to(device=device)

    return idt, t_tensor, TY_tensor


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
    species_names = np.array(['0', '1', '3', '6', '7', '8', '11', '14', '16', '17'])
    species_names_int = np.array([0, 1, 3, 6, 7, 8, 11, 14, 16, 17])
    fig = plt.figure(figsize=(27, 24))
    for i in range(len(species_names)):
        i_species = i

        ax = fig.add_subplot(4, 6, i + 1)
        ax.plot(x_pinn / x_scale,
                y_pinn[:, i_species],
                color="r",
                ls="-",
                label='pinn')

        ax.plot(t_true / x_scale,
                y_true[:, species_names_int[i]],
                color="b",
                ls="-",
                marker=None,
                fillstyle='none',
                markersize=6,
                alpha=1.0,
                label='true')

        #ax.set_xlim(left=1e-2)
        #ax.set_xscale('log')

        ax2 = ax.twinx()
        ax2.plot(x_pinn / x_scale,
                 dydt[:, i_species],
                 ls='--',
                 color='r',
                 alpha=0.8,
                 label='dydt')

        ax2.plot(x_pinn / x_scale,
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

    ax = fig.add_subplot(4, 6, i + 2)
    epoch = len(loss_list['res_train'])
    for key in key_list_loss:
        ax.plot(np.arange(epoch) + 1, loss_list[key], '-', lw=2, label=key)
    ax.legend(loc='upper right', shadow=False, framealpha=0.3)
    ax.set_yscale("log")
    if epoch > 2000:
        ax.set_xscale("log")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    key_list_loss2 = ['res_0', 'res_1', 'res_3', 'res_6', 'res_7',
                      'res_8', 'res_11', 'res_14', 'res_16', 'res_17']

    ax = fig.add_subplot(4, 6, i + 3)
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

    ax = fig.add_subplot(4, 6, i + 4)
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

    ax = fig.add_subplot(4, 6, i + 5)
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

    # plt.show()

    figname = 'figs/pinn_loss.png'
    fig.savefig(figname, dpi=120, facecolor='w', edgecolor='w', orientation='portrait')

    ax.cla()
    fig.clf()
    plt.close()
