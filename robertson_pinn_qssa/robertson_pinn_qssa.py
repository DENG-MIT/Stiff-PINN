import numpy as np
import matplotlib.pyplot as plt
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _utils import  grad_norm, grad_mean, plot_pinn_y, to_np
from config import cuda_index, device
from pinn_model import MyDataSet, PINN_Model

torch.set_default_tensor_type("torch.DoubleTensor")

# First calculate by Datagen_robertson

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


def get_solution(y0, t_end, n_steps, t_np):
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


ode = ODE(3)

n_exp = 1
y_list = []
dydt_list = []
np.random.seed(0)

t_end = 1e5
n_steps = 100

t_np = np.logspace(start=-2, stop=5, num=n_steps, endpoint=True)

n_steps = t_np.shape[0]

for i in range(n_exp):
    #y0 = np.random.uniform(np.array([0.5, 0, 0]), np.array([1.5, 0, 0]))
    y0 = np.array([1, 0, 0])
    print(i, y0)
    y = get_solution(y0, t_end, n_steps, t_np)
    y_list.append(y[1:, :])

y_np = np.vstack(y_list)

#np.savez('./Datasets/Robertson_PINN Dataset.npz', t=t_np, y=y_np)

# print(t_np)
# print(y_np)

print(type(t_np))
print(type(y_np))

print(t_np.shape, y_np.shape)

for i_exp in range(n_exp):
    fig = plt.figure(figsize=(9, 8))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(t_np, y_list[i_exp][:, i], label='y_{}'.format(i + 1))
        plt.ylabel('State')
        plt.xscale('log')
        plt.xlim(1e-6, 1e5)
        plt.legend()
    plt.xlabel('Time')
    plt.title('exp {}'.format(i_exp))
    fig.tight_layout()
    plt.savefig('./figs/true_y_exp_{}'.format(i_exp))
    # plt.show()

# Then calculate by pinn
is_restart = True
n_grid_train = 2500
n_grid_test = 100
learning_rate = 1e-3

batch_size_pinn = 2500
num_epochs = 30000
printout_freq = 50
model_checking_freq = 10

alpha_lambda = 0.1

n_var = 2

t_true = torch.from_numpy(t_np)
t_true.to(device=device)
y_true = torch.from_numpy(y_np)
y_true.to(device=device)

# idt = t_true.max(dim=0).values
idt = 1e5
x_scale = idt
y_scale_old = y_true.max(dim=0).values.to(device=device)
y_scale = torch.Tensor([y_scale_old[0], y_scale_old[2]]).to(device=device)
w_scale = torch.ones(n_var).to(device=device) * y_scale

checkpoint_path = 'models/robertson_pinn.pt'

y0 = torch.Tensor([[1, 0]]).to(device=device)

net = PINN_Model(nodes=120, layers=3, y0=y0, w_scale=w_scale,
                 x_scale=x_scale).to(device=device)
net.xavier_init()

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                             weight_decay=4e-5)

loss_list = {}
key_list_loss = ['res_train', 'res_test', 'res_0', 'res_2', 'alpha0',
                 'alpha2', 'grad_norm', 'slope']
key_list_alpha = []
for key in key_list_loss + key_list_alpha:
    loss_list[key] = []

epoch_old = 0

# prepare data
t_end = t_true.max().item()
print("t_end")
print("{}".format(t_end))
eps = 1e-30

if is_restart is True:
    checkpoint = torch.load(checkpoint_path + '.tar', map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_old = checkpoint['epoch']
    loss_list = checkpoint['loss_list']
    x_train_all = torch.Tensor(checkpoint['x_train']).to(device)
    x_test_all = torch.Tensor(checkpoint['x_test']).to(device)

grad_mean_res = torch.ones(n_var)
alpha_res = torch.ones(n_var)

# minibatch
pinn_t_list = torch.logspace(start=-2, end=np.log10(t_end),
                             steps=n_grid_train,
                             requires_grad=False).unsqueeze(dim=1).to(device=device)
#pinn_t_list2 = torch.logspace(start=4, end=np.log10(t_end),
                             #steps=2000,
                             #requires_grad=False).unsqueeze(dim=1).to(device=device)
#pinn_t_list = torch.cat([pinn_t_list1,pinn_t_list2],dim=0)

# make PyTorch dataset
pinn_data = MyDataSet(data=pinn_t_list, label=pinn_t_list)
pinn_loader = DataLoader(pinn_data, batch_size=batch_size_pinn, shuffle=False, drop_last=True, pin_memory=False)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=1, epochs=num_epochs)

x_test = torch.logspace(start=-2, end=np.log10(t_end),
                                steps=n_grid_test,
                                requires_grad=False).unsqueeze(dim=1).to(device=device)

for epoch in tqdm(range(num_epochs)):
    if is_restart:
        if epoch < epoch_old:
            continue

    i_sample = 0
    for i_sample, (x_train, _) in enumerate(pinn_loader):

        x_all = torch.cat([x_train, x_test], dim=0)
        x_all.requires_grad = True

        y_all = net(x_all).abs()

        rhs_all = torch.empty_like(y_all)
        dydt_all = torch.empty_like(rhs_all)

        for i in range(dydt_all.shape[1]):
            dydt_all[:, i] = torch.autograd.grad(outputs=y_all[:, i].sum(),
                                                 inputs=x_all,
                                                 retain_graph=True,
                                                 create_graph=True,
                                                 allow_unused=True)[0].view(-1)

        B = torch.ones(y_all.shape[0]).to(device=device)

        delta = K[2] * K[2] * y_all[:, 1] * y_all[:, 1] + 4 * K[0] * y_all[:, 0] * K[1]
        B[:] = (-K[2] * y_all[:, 1] + torch.sqrt(delta)) / (2 * K[1])
        rhs_all[:, 0] = -K[0] * y_all[:, 0] + K[2] * B[:] * y_all[:, 1]
        rhs_all[:, 1] = K[1] * B[:] * B[:]

        y_train = y_all[:batch_size_pinn, :]
        y_test = y_all[batch_size_pinn:, :]
        rhs_train = rhs_all[:batch_size_pinn, :]
        rhs_test = rhs_all[batch_size_pinn:, :]
        dydt_train = dydt_all[:batch_size_pinn, :]
        dydt_test = dydt_all[batch_size_pinn:, :]

        loss_res_train0 = criterion(dydt_train[:, 0], rhs_train[:, 0])
        loss_res_train1 = criterion(dydt_train[:, 1], rhs_train[:, 1])

        loss_res_test0 = criterion(dydt_test[:, 0], rhs_test[:, 0])
        loss_res_test1 = criterion(dydt_test[:, 1], rhs_test[:, 1])

        loss_train = (loss_res_train0 + loss_res_train1) * 1e5
        loss_test = (loss_res_test0 + loss_res_test1) * 1e5

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        grad_sum = grad_norm(net)
        slope = 1.0 #net.get_slope()

    loss_list['res_train'].append(loss_train.item())
    loss_list['res_test'].append(loss_test.item())
    loss_list['res_0'].append(loss_res_train0.item())
    loss_list['res_2'].append(loss_res_train1.item())
    loss_list['alpha0'].append(alpha_res[0].item())
    loss_list['alpha2'].append(alpha_res[1].item())
    loss_list['slope'].append(slope)
    loss_list['grad_norm'].append(grad_sum)

    if epoch % printout_freq == 0:
        print('\n @epoch {} cuda {} slope {:.2f} grad_norm {:.2e}'.format(
            epoch, cuda_index, slope, grad_sum))
        print(['alpha {} = {:.2e}'.format(key, loss_list[key][epoch]) for key in key_list_alpha])
        print(['{} = {:.2e}'.format(key, loss_list[key][epoch]) for key in key_list_loss])

        sorted_index = x_train.sort(dim=0).indices.view(-1)

        # plot here
        plot_pinn_y(to_np(x_train[sorted_index]),
                    to_np(y_train[sorted_index]),
                    to_np(t_true),
                    to_np(y_true),
                    to_np(dydt_train[sorted_index]),
                    to_np(rhs_train[sorted_index]),
                    loss_list,
                    1)

        torch.save({'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_list': loss_list,
                    'x_train': to_np(x_train),
                    'x_test': to_np(x_test),
                    }, checkpoint_path + '.tar')

