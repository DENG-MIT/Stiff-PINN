import numpy as np
import matplotlib.pyplot as plt
from assimulo.solvers import ExplicitEuler,CVode
from assimulo.problem import Explicit_Problem

K = [0.04, 3.E7, 1.E4]


class ODE_QSSA():
    def __init__(self, species_count):
        self.species_count = species_count
        self.dx_dt = np.zeros(species_count)
        
        self.coeffs = [- 0.5 * K[2] / K[1], 0.25 * K[2] **2 / (K[1] ** 2), \
                       K[0] / K[1]]
        
    def __call__(self, t, x):
        
        # x[0]: x; x[1]: z
        
        self.y = self.coeffs[0] * x[1] + np.sqrt(self.coeffs[1] * x[1] * x[1] \
                            + self.coeffs[2] * x[0] )
        
        self.dx_dt[0] = -K[0] * x[0] + K[2] * self.y * x[1]
        self.dx_dt[1] = K[1] * self.y * self.y

        return self.dx_dt
    

class ODE():
    def __init__(self, species_count):
        self.species_count = species_count
        self.dx_dt = np.zeros(species_count)

    def __call__(self, t, x):

        self.dx_dt[0] = -K[0] * x[0] + K[2] * x[1] * x[2]
        self.dx_dt[1] = K[0] * x[0] - K[1] * x[1] ** 2 - K[2] * x[1] * x[2]
        self.dx_dt[2] = K[1] * x[1] ** 2

        return self.dx_dt

def get_solution(ode, y0, t_end, n_steps, t_np, solver, atol):
    '''Use solve_ivp from scipy to solve the ODE'''

    exp_mod = Explicit_Problem(
        ode, y0, name='Robertson Chemical Kinetics Example')
    
    if solver == 'Euler':
        # Create an Assimulo explicit solver (Euler)
        exp_sim = ExplicitEuler(exp_mod)
        
    elif solver == 'CVode':
        exp_sim = CVode(exp_mod)
    
        #Sets the solver paramters
        exp_sim.iter = 'Newton'
        exp_sim.discr = 'BDF'
        exp_sim.rtol = 1.e-4
        exp_sim.atol = atol
    
    else:
        print('Invalid solver')
        return 
        
    # Simulate
    # ncp = 0 will print the internal time step
    t, y = exp_sim.simulate(tfinal=t_end,ncp=0, ncp_list=t_np)

    return t,y


ode_qssa = ODE_QSSA(2)

ode = ODE(3)

n_exp = 1
y_list = []
dydt_list = []
np.random.seed(0)

t_end = 1e5
n_steps = 100

t_np = np.logspace(start=-6, stop=5, num=n_steps, endpoint=True)

n_steps = t_np.shape[0]

for i in range(n_exp):
    y0_qssa = np.random.uniform(np.array([0.5, 0]), np.array([1.5, 0]))
    y0 = np.random.uniform(np.array([0.5, 0, 0]), np.array([1.5, 0, 0]))
    y0[0:1] = y0_qssa[0:1]
    print(i, y0)
    t1,y1 = get_solution(ode_qssa, y0_qssa, t_end, n_steps, t_np,'CVode',\
                         np.array([1.0e-8, 1.0e-14]))
    t2,y2 = get_solution(ode_qssa, y0_qssa, t_end, n_steps, t_np,'Euler',\
                         np.array([1.0e-8, 1.0e-14]))
    #t2, y2 = get_solution(ode, y0, t_end, n_steps, t_np,'CVode',\
    #                      np.array([1.0e-8, 1.0e-14, 1.0e-6]))

#np.savez('./Datasets/Robertson_QSSA_CVode.npz', t=t1, y=y1)
#np.savez('./Datasets/Robertson_QSSA_Euler.npz', t=t2, y=y2)
#np.savez('./Datasets/Robertson_CVode.npz', t=t2, y=y2)

for i_exp in range(n_exp):
    fig = plt.figure(figsize=(9, 8))
    index_orig = [0, 2]
    for i in range(2):
        plt.subplot(2, 1, i+1)
        plt.plot(t1, y1[:,i], label='y_{}_QSSA_CVode'.format(i+1))
        plt.plot(t2, y2[:,i], label='y_{}_QSSA_Euler'.format(i+1))
        plt.ylabel('State')
        plt.xscale('log')
        plt.xlim(1e-6, 1e5)
        plt.legend()
    plt.xlabel('Time')
    plt.title('exp {}'.format(i_exp))
    fig.tight_layout()
    plt.savefig('./figs/y_qssa_compare_CVode_Euler'.format(i_exp))
    plt.show()


