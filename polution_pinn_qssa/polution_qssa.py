import numpy as np
import matplotlib.pyplot as plt
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _utils import get_idt, grad_norm, grad_mean, plot_pinn_y, to_np
from config import cuda_index, device
from pinn_model import MyDataSet, PINN_Model

torch.set_default_tensor_type("torch.DoubleTensor")

# First calculate by Datagen_robertson

K = [0.350, 0.266E2, 0.123E5, 0.860E-3, 0.820E-3, 0.150E5, 0.130E-3, 0.240E5,
     0.165E5, 0.900E4, 0.220E-1, 0.120E5, 0.188E1, 0.163E5, 0.480E7, 0.350E-3,
     0.175E-1, 0.100E9, 0.444E12, 0.124E4, 0.210E1, 0.578E1, 0.474E-1, 0.178E4, 0.312E1]


class ODE():
    def __init__(self, species_count):
        self.species_count = species_count
        self.dx_dt = np.zeros(species_count)

    def __call__(self, t, x):
        y10 = K[7] * x[5] * (2 * K[15] * K[17] * x[2] / (K[17] + K[18]) + 2 * K[3] * x[3] + 2 * K[6] * x[5]) / (K[8] * K[13] * x[0] * x[1] + 1e-4)
        y5 = K[8] * x[1] * y10 / (K[7] * x[5])
        y2 = (K[0] * x[0] + K[16] * x[2] + K[15] * K[18] * x[2] / (K[17] + K[18]) + \
              K[21] * K[22] * x[0] * x[2] / (K[20] + K[21])) / K[14]
        y4 = ((K[5] * x[3] + K[7] * x[5] + K[13] * x[0] + K[19] * x[8]) * y5 \
              - 2 * K[17] * K[15] * x[2] / (K[17] + K[18])) / (K[2] * x[1])
        y9 = (K[6] * x[5] + K[8] * y10 * x[1]) / (K[11] * x[1])
        y12 = K[9] * y10 * x[0] / K[10]
        y13 = (K[6] * x[5] + K[8] * y10 * x[1]) / K[12]
        y15 = K[15] * x[2] / (K[17] + K[18])
        y18 = K[22] * x[0] * x[2] / (K[20] + K[21])
        y19 = K[22] * K[23] * x[0] * x[0] * x[2] / (K[24] * (K[20] + K[21]))

        self.dx_dt[0] = -K[0] * x[0] - K[9] * y10 * x[0] - K[13] * x[0] * y5 \
                        - K[22] * x[0] * x[2] - K[23] * y18 * x[0] + \
                        K[1] * x[1] * x[2] + K[2] * y4 * x[1] + K[8] * y10 * x[1] \
                        + K[10] * y12 + K[11] * y9 * x[1] + K[21] * y18 + K[24] * y19
        self.dx_dt[1] = -K[1] * x[1] * x[2] - K[2] * y4 * x[1] - K[8] * y10 * x[1] \
                        - K[11] * y9 * x[1] + K[0] * x[0] + K[20] * y18
        self.dx_dt[2] = -K[1] * x[1] * x[2] - K[15] * x[2] - K[16] * x[2] \
                        - K[22] * x[0] * x[2] + K[14] * y2
        self.dx_dt[3] = -K[3] * x[3] - K[4] * x[3] - K[5] * x[3] * y5 + K[12] * y13
        self.dx_dt[4] = K[3] * x[3] + K[4] * x[3] + K[5] * x[3] * y5 + K[6] * x[5]
        self.dx_dt[5] = -K[6] * x[5] - K[7] * x[5] * y5
        self.dx_dt[6] = K[8] * y10 * x[1]
        self.dx_dt[7] = K[13] * x[0] * y5
        self.dx_dt[8] = -K[19] * x[8] * y5
        self.dx_dt[9] = K[19] * x[8] * y5

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
    exp_sim.rtol = 1.e-10
    #exp_sim.atol = np.array([1.0e-10, 1.0e-9, 1.0e-17, 1.0e-10, 1.0e-15, 1.0e-15,
    #                         1.0e-9, 1.0e-9, 1.0e-11, 1.0e-16, 1.0e-16, 1.0e-11,
    #                         1.0e-12, 1.0e-14, 1.0e-11, 1.0e-27, 1.0e-11, 1.0e-14, 1.0e-14, 1.0e-14])
    exp_sim.atol = np.array([1.0e-10, 1.0e-9, 1.0e-10, 1.0e-9, 1.0e-9, 1.0e-11, 1.0e-11, 1.0e-11, 1.0e-11, 1.0e-14])
    #exp_sim.atol = np.ones(20) * 1e-10

    # Simulate
    # ncp = 0 will print the internal time step
    t, y = exp_sim.simulate(tfinal=t_end, ncp=0, ncp_list=t_np)

    return y


ode = ODE(10)

n_exp = 1
y_list = []
dydt_list = []
np.random.seed(0)

t_end = 60
n_steps = 100000

t_np = np.linspace(start=0.01, stop=60, num=n_steps, endpoint=True)
n_steps = t_np.shape[0]

for i in range(n_exp):
    y0 = np.array([0, 0.2, 0.04, 0.1, 0.3, 0.01, 0, 0, 0.007, 0])
    y = get_solution(y0, t_end, n_steps, t_np)
    y_list.append(y[1:, :])

y_np = np.vstack(y_list)
for i in range(10):
    print("{}".format(y_list[0][99999, i]))

for i_exp in range(n_exp):
    fig = plt.figure(figsize=(9, 8))
    for i in range(10):
        plt.subplot(4, 6, i + 1)
        plt.plot(t_np, y_list[i_exp][:, i], label='y_{}'.format(i + 1))
        plt.ylabel('State')
        #plt.xscale('log')
        plt.xlim(0, 60)
        plt.legend()
    plt.xlabel('Time')
    plt.title('exp {}'.format(i_exp))
    fig.tight_layout()
    plt.savefig('./figs/true_y_exp_{}'.format(i_exp))


# Then calculate by pinn
is_restart = False
n_grid_train = 500
n_grid_test = 100
learning_rate = 1e-3

batch_size_pinn = 500
num_epochs = 200000
printout_freq = 500
model_checking_freq = 10

n_var = 10
slowindex = [0, 1, 3, 6, 7, 8, 11, 14, 16, 17]

t_true = torch.from_numpy(t_np)
t_true.to(device=device)
y_true = torch.from_numpy(y_np)
y_true.to(device=device)

# idt = t_true.max(dim=0).values
idt = 60
x_scale = idt
y_scale_old = y_true.max(dim=0).values.to(device=device)
y_scale = torch.Tensor(y_scale_old).to(device=device)
w_scale = torch.ones(n_var).to(device=device) * y_scale

checkpoint_path = 'models/robertson_pinn.pt'

y0 = torch.Tensor([[0, 0.2, 0.04, 0.1, 0.3, 0.01, 0, 0, 0.007, 0]]).to(device=device)

net = PINN_Model(nodes=120, layers=7, y0=y0, w_scale=w_scale,
                 x_scale=x_scale).to(device=device)
net.xavier_init()

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                             weight_decay=1e-6)

loss_list = {}
key_list_loss = ['res_train', 'res_test', 'res_0', 'res_1', 'res_3', 'res_6', 'res_7',
                 'res_8', 'res_11', 'res_14', 'res_16', 'res_17', 'grad_norm', 'slope']
key_list_alpha = []
for key in key_list_loss + key_list_alpha:
    loss_list[key] = []

epoch_old = 0

# prepare data
t_end = t_true.max().item()
print("t_end")
print("{}".format(t_end))
eps = 1e-30


pinn_t_list = torch.linspace(start=0, end=t_end,
                             steps=n_grid_train,
                             requires_grad=False).unsqueeze(dim=1).to(device=device)
x_test = torch.linspace(start=0, end=t_end,
                                steps=n_grid_test,
                                requires_grad=False).unsqueeze(dim=1).to(device=device)

#pinn_t_list = torch.logspace(start=-2, end=np.log10(t_end),
#                             steps=n_grid_train,
#                             requires_grad=False).unsqueeze(dim=1).to(device=device)
#x_test = torch.logspace(start=-2, end=np.log10(t_end),
#                                steps=n_grid_test,
#                                requires_grad=False).unsqueeze(dim=1).to(device=device)

if is_restart is True:
    checkpoint = torch.load(checkpoint_path + '.tar', map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_old = checkpoint['epoch']
    loss_list = checkpoint['loss_list']
    pinn_t_list = torch.Tensor(checkpoint['x_train']).to(device)
    x_test = torch.Tensor(checkpoint['x_test']).to(device)

grad_mean_res = torch.ones(n_var)
alpha_res = torch.ones(n_var)

## minibatch
# make PyTorch dataset
pinn_data = MyDataSet(data=pinn_t_list, label=pinn_t_list)
pinn_loader = DataLoader(pinn_data, batch_size=batch_size_pinn, shuffle=True, drop_last=True, pin_memory=False)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=1, epochs=num_epochs)

for epoch in tqdm(range(num_epochs)):
    if is_restart:
        if epoch < epoch_old:
            continue

    i_sample = 0
    for i_sample, (x_train, _) in enumerate(pinn_loader):
        x_all = torch.cat([x_train, x_test], dim=0)
        x_all.requires_grad = True

        x_all = x_all + 0.01
        y_all = net(x_all).abs()
        rhs_all = torch.empty_like(y_all)
        dydt_all = torch.empty_like(rhs_all)

        for i in range(dydt_all.shape[1]):
            dydt_all[:, i] = torch.autograd.grad(outputs=y_all[:, i].sum(),
                                                 inputs=x_all,
                                                 retain_graph=True,
                                                 create_graph=True,
                                                 allow_unused=True)[0].view(-1)



        y10 = K[7] * y_all[:, 5] * (2 * K[15] * K[17] * y_all[:, 2] / (K[17] + K[18]) + 2 * K[3] * y_all[:, 3] + \
                                    2 * K[6] * y_all[:, 5]) / (K[8] * K[13] * y_all[:, 0] * y_all[:, 1])
        y5 = K[8] * y_all[:, 1] * y10[:] / (K[7] * y_all[:, 5])
        y2 = (K[0] * y_all[:, 0] + K[16] * y_all[:, 2] + K[15] * K[18] * y_all[:, 2] / (K[17] + K[18]) + \
              K[21] * K[22] * y_all[:, 0] * y_all[:, 2] / (K[20] + K[21])) / K[14]
        y4 = ((K[5] * y_all[:, 3] + K[7] * y_all[:, 5] + K[13] * y_all[:, 0] + K[19] * y_all[:, 8]) * y5[:] \
              - 2 * K[17] * K[15] * y_all[:, 2] / (K[17] + K[18])) / (K[2] * y_all[:, 1])
        y9 = (K[6] * y_all[:, 5] + K[8] * y10[:] * y_all[:, 1]) / (K[11] * y_all[:, 1])
        y12 = K[9] * y10[:] * y_all[:, 0] / K[10]
        y13 = (K[6] * y_all[:, 5] + K[8] * y10[:] * y_all[:, 1]) / K[12]
        y15 = K[15] * y_all[:, 2] / (K[17] + K[18])
        y18 = K[22] * y_all[:, 0] * y_all[:, 2] / (K[20] + K[21])
        y19 = K[22] * K[23] * y_all[:, 0] * y_all[:, 0] * y_all[:, 2] / (K[24] * (K[20] + K[21]))

        rhs_all[:, 0] = -K[0] * y_all[:, 0] - K[9] * y10[:] * y_all[:, 0] - K[13] * y_all[:, 0] * y5[:] \
                        - K[22] * y_all[:, 0] * y_all[:, 2] - K[23] * y18[:] * y_all[:, 0] + \
                        K[1] * y_all[:, 1] * y_all[:, 2] + K[2] * y4[:] * y_all[:, 1] + K[8] * y10[:] * y_all[:, 1] \
                        + K[10] * y12[:] + K[11] * y9[:] * y_all[:, 1] + K[21] * y18[:] + K[24] * y19[:]
        rhs_all[:, 1] = -K[1] * y_all[:, 1] * y_all[:, 2] - K[2] * y4[:] * y_all[:, 1] - K[8] * y10[:] * y_all[:, 1] \
                        - K[11] * y9[:] * y_all[:, 1] + K[0] * y_all[:, 0] + K[20] * y18[:]
        rhs_all[:, 2] = -K[1] * y_all[:, 1] * y_all[:, 2] - K[15] * y_all[:, 2] - K[16] * y_all[:, 2] \
                        - K[22] * y_all[:, 0] * y_all[:, 2] + K[14] * y2[:]
        rhs_all[:, 3] = -K[3] * y_all[:, 3] - K[4] * y_all[:, 3] - K[5] * y_all[:, 3] * y5[:] + K[12] * y13[:]
        rhs_all[:, 4] = K[3] * y_all[:, 3] + K[4] * y_all[:, 3] + K[5] * y_all[:, 3] * y5[:] + K[6] * y_all[:, 5]
        rhs_all[:, 5] = -K[6] * y_all[:, 5] - K[7] * y_all[:, 5] * y5[:]
        rhs_all[:, 6] = K[8] * y10[:] * y_all[:, 1]
        rhs_all[:, 7] = K[13] * y_all[:, 0] * y5[:]
        rhs_all[:, 8] = -K[19] * y_all[:, 8] * y5[:]
        rhs_all[:, 9] = K[19] * y_all[:, 8] * y5[:]


        y_train = y_all[:batch_size_pinn, :]
        y_test = y_all[batch_size_pinn:, :]
        rhs_train = rhs_all[:batch_size_pinn, :]
        rhs_test = rhs_all[batch_size_pinn:, :]
        dydt_train = dydt_all[:batch_size_pinn, :]
        dydt_test = dydt_all[batch_size_pinn:, :]

        loss_res_train0 = criterion(dydt_train[:, 0], rhs_train[:, 0])
        loss_res_train1 = criterion(dydt_train[:, 1], rhs_train[:, 1])
        loss_res_train2 = criterion(dydt_train[:, 2], rhs_train[:, 2])
        loss_res_train3 = criterion(dydt_train[:, 3], rhs_train[:, 3])
        loss_res_train4 = criterion(dydt_train[:, 4], rhs_train[:, 4])
        loss_res_train5 = criterion(dydt_train[:, 5], rhs_train[:, 5])
        loss_res_train6 = criterion(dydt_train[:, 6], rhs_train[:, 6])
        loss_res_train7 = criterion(dydt_train[:, 7], rhs_train[:, 7])
        loss_res_train8 = criterion(dydt_train[:, 8], rhs_train[:, 8])
        loss_res_train9 = criterion(dydt_train[:, 9], rhs_train[:, 9])

        loss_res_test0 = criterion(dydt_test[:, 0], rhs_test[:, 0])
        loss_res_test1 = criterion(dydt_test[:, 1], rhs_test[:, 1])
        loss_res_test2 = criterion(dydt_test[:, 2], rhs_test[:, 2])
        loss_res_test3 = criterion(dydt_test[:, 3], rhs_test[:, 3])
        loss_res_test4 = criterion(dydt_test[:, 4], rhs_test[:, 4])
        loss_res_test5 = criterion(dydt_test[:, 5], rhs_test[:, 5])
        loss_res_test6 = criterion(dydt_test[:, 6], rhs_test[:, 6])
        loss_res_test7 = criterion(dydt_test[:, 7], rhs_test[:, 7])
        loss_res_test8 = criterion(dydt_test[:, 8], rhs_test[:, 8])
        loss_res_test9 = criterion(dydt_test[:, 9], rhs_test[:, 9])

        loss_train = (loss_res_train0 + loss_res_train1 + loss_res_train2 + loss_res_train3 + loss_res_train4 + \
                      loss_res_train5 * 10 + loss_res_train6 * 10 + loss_res_train7 * 10 + loss_res_train8 * 10 + loss_res_train9 * 100) * 1e6
        loss_test = (loss_res_test0 + loss_res_test1 + loss_res_test2 + loss_res_test3 + loss_res_test4 + \
                     loss_res_test5 * 10 + loss_res_test6 * 10 + loss_res_test7 * 10 + loss_res_test8 * 10 + loss_res_test9 * 100) * 1e6

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        grad_sum = grad_norm(net)
        slope = net.get_slope()

    loss_list['res_train'].append(loss_train.item())
    loss_list['res_test'].append(loss_test.item())
    loss_list['res_0'].append(loss_res_train0.item())
    loss_list['res_1'].append(loss_res_train1.item())
    loss_list['res_3'].append(loss_res_train2.item())
    loss_list['res_6'].append(loss_res_train3.item())
    loss_list['res_7'].append(loss_res_train4.item())
    loss_list['res_8'].append(loss_res_train5.item())
    loss_list['res_11'].append(loss_res_train6.item())
    loss_list['res_14'].append(loss_res_train7.item())
    loss_list['res_16'].append(loss_res_train8.item())
    loss_list['res_17'].append(loss_res_train9.item())
    loss_list['slope'].append(slope)
    loss_list['grad_norm'].append(grad_sum)

    if epoch % printout_freq == 0:
        print('\n @epoch {} cuda {} slope {:.2f} grad_norm {:.2e}'.format(
            epoch, cuda_index, slope, grad_sum))
        print(['alpha {} = {:.2e}'.format(key, loss_list[key][epoch]) for key in key_list_alpha])
        print(['{} = {:.2e}'.format(key, loss_list[key][epoch]) for key in key_list_loss])

        sorted_index = x_train.sort(dim=0).indices.view(-1)

        # plot here
        plot_pinn_y(to_np(x_train[sorted_index].clamp(1e-3)),
                    to_np(y_train[sorted_index]),
                    to_np(t_true.clamp(1e-3)),
                    to_np(y_true),
                    to_np(dydt_train[sorted_index]),
                    to_np(rhs_train[sorted_index]),
                    loss_list,
                    x_scale)

        torch.save({'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_list': loss_list,
                    'x_train': to_np(pinn_t_list),
                    'x_test': to_np(x_test),
                    }, checkpoint_path + '.tar')

        torch.save(net, checkpoint_path)