import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
torch.set_default_tensor_type('torch.DoubleTensor')

# Convex case, \hat{f}(x,c) = Y_1(x)A + Y_2(x)\alpha(c)
class Pendulum:

    def __init__(self, Wind, Noise, init_theta=0.0, init_dtheta=0.0, duration=20, Cd=0.3, \
                 alpha=0.5, mode='baseline', eta1_base=0.01, eta2_base=0.01):
        self.duration = duration
        
        # Pendulum parameters
        self.m = 1.0
        self.l = 0.5
        self.g = 9.81                               
        self.g_hat = 9.0
        
        # States                 
        self.theta = init_theta                   
        self.dtheta = init_dtheta                                                               
        self.state = np.array([self.theta, self.dtheta])
        self.prev_state = self.state
        self.dstate = np.array([0.0, 0.0])
        self.u = 0.0
        
        # Control gain
        self.gain = 1.5
        
        # Noise
        self.Noise = Noise
        self.a_noise = 0.0
        
        # Step and meta step
        self.step_size = 1e-2
        self.total_step = 0 
        self.total_wind = 0
        self.sub_step = 0
        self.trigger = False
    
        # Wind condition
        self.Wind = Wind
        self.Wind_x = self.Wind[self.total_wind][0]
        self.Wind_y = self.Wind[self.total_wind][1]
        self.Cd = Cd
        self.alpha = alpha
    
        # Fd gt and measurement
        self.F_d_data = 0.0
        self.F_d_gt = 0.0
        self.F_d_hat = 0.0
    
        # Learning method
        self.mode = mode
        
        # Kernel initialization 
        self.dim_y1 = 30
        self.dim_y2 = 20
        self.w1 = np.random.normal(size=(self.dim_y1, 2))
        self.w2 = np.random.normal(size=(self.dim_y2, 2))
        self.b1 = np.random.uniform(low=0, high=2*np.pi, size=self.dim_y1)
        self.b2 = np.random.uniform(low=0, high=2*np.pi, size=self.dim_y2)
        self.a1 = np.zeros(self.dim_y1)
        self.a2 = np.zeros(self.dim_y2)
        self.y1 = 0.0
        self.y2 = 0.0
        self.y = 0.0
        
        self.y2_batch = [] # for meta-adaptation
        self.F_d_data_batch = [] # for meta-adaptation
        self.state_batch = [] # for meta-adaptation
    
        # Learning rate
        self.eta1 = 0.0
        self.eta2 = 0.0
        self.eta1_base = eta1_base
        self.eta2_base = eta2_base
    
    # Ground truth damping function.
    def F_d(self):
        # External wind velocity
        w_x = self.Wind_x
        w_y = self.Wind_y
        v_x = self.l * self.dtheta * np.cos(self.theta)
        v_y = self.l * self.dtheta * np.sin(self.theta)
        R = np.array([w_x - v_x, w_y - v_y])
        F = self.Cd * np.linalg.norm(R) * R
        return -(self.l * np.sin(self.theta) * F[1] + self.l * np.cos(self.theta) * F[0]) \
               - self.alpha*self.dtheta
    
    def kernel(self):
        self.y1 = np.dot(np.cos(self.w1 @ self.state + self.b1), self.a1)
        self.y2 = np.dot(np.cos(self.w2 @ self.state + self.b2), self.a2)
        self.y = self.y1 + self.y2
        return self.y
    
    def noise(self): 
        self.a_noise = self.Noise[self.total_step]
    
    def F_hat(self):
        if self.mode == 'oracle':
            self.F_d_hat = self.F_d()
        elif self.mode == 'baseline':
            self.F_d_hat = 0.0
        elif self.mode == 'adapt':
            self.F_d_hat = self.kernel()
        elif self.mode == 'meta-adapt':
            self.F_d_hat = self.kernel()
        else:
            raise Exception('Not Implemented!')
        return self.F_d_hat
    
    def controller(self):
        #(des_theta, des_dtheta, des_ddtheta) = self.desired_trajectory()
        u_feedback = self.m * self.l**2 * (-2 * self.gain * self.dtheta - self.gain**2 * self.theta)
        u_feedforward = -self.m * self.l * self.g_hat * np.sin(self.theta)
        u_residual = -self.F_hat()
        self.u = u_feedback + u_feedforward + u_residual
    
    def adapt(self):
        self.eta1 = self.eta1_base / np.sqrt(self.sub_step)
        self.a1 -= self.eta1 * (self.y-self.F_d_data) * np.cos(self.w1 @ self.prev_state + self.b1)
    
        self.eta2 = self.eta2_base / np.sqrt(self.sub_step)
        self.a2 -= self.eta2 * (self.y-self.F_d_data) * np.cos(self.w2 @ self.prev_state + self.b2)
    
    def meta_adapt(self):
        self.eta2 = self.eta2_base / np.sqrt(self.sub_step)
        self.a2 -= self.eta2 * (self.y-self.F_d_data) * np.cos(self.w2 @ self.prev_state + self.b2)
        
        self.y2_batch.append(np.copy(self.y2))
        self.F_d_data_batch.append(np.copy(self.F_d_data))
        self.state_batch.append(np.copy(self.prev_state))
        
        if self.trigger:
            self.eta1 = self.eta1_base / np.sqrt(self.total_wind)
            a1_grad = np.zeros_like(self.a1)
            for i in range(len(self.y2_batch)):
                y2 = self.y2_batch[i]
                f = self.F_d_data_batch[i]
                s = self.state_batch[i]
                y1 = np.dot(np.cos(self.w1 @ s + self.b1), self.a1)
                a1_grad += (y1 + y2 - f) * np.cos(self.w1 @ s + self.b1)
            self.a1 -= self.eta1 * a1_grad
            self.trigger = False
            self.y2_batch = []
            self.F_d_data_batch = []
            self.state_batch = []
    
    def dynamics(self):
        self.dstate[0] = self.dtheta
        self.dstate[1] = self.u/(self.m*self.l**2) + self.g/self.l*np.sin(self.theta) + self.F_d()/(self.m*self.l**2)
        self.dstate[1] += self.a_noise
        
    # ODE solver: (4,5) Runge-Kutta
    def process(self):
        self.noise()
        self.controller()
        self.F_d_gt = self.F_d()
        prev_state = self.state
        self.prev_state = prev_state
        
        self.dynamics()
        s1_dstate = self.dstate
        
        self.state = prev_state + 0.5 * self.step_size * s1_dstate
        self.dynamics()
        s2_dstate = self.dstate
        
        self.state = prev_state + 0.5 * self.step_size * s2_dstate
        self.dynamics()
        s3_dstate = self.dstate
        
        self.state = prev_state + self.step_size * s3_dstate
        self.dynamics()
        s4_dstate = self.dstate
        
        self.state = prev_state + 1.0 / 6 * self.step_size * \
                      (s1_dstate + 2 * s2_dstate + 2 * s3_dstate + s4_dstate)
        
        self.total_step += 1
        self.sub_step += 1
        
        self.dtheta = self.state[1]
        self.theta = self.state[0]
        
        # F_d measurement
        # self.F_d_data = -self.u - self.m*self.g_hat*self.l*np.sin(self.theta) + self.m*self.l**2*(self.state[1]-prev_state[1])/self.step_size
        self.F_d_data = self.F_d_gt + self.a_noise
        if self.mode == 'adapt':
            self.adapt()
        if self.mode == 'meta-adapt':
            self.meta_adapt()
        
    def simulate(self):
        Theta = []
        Theta = np.append(Theta, self.theta)
        Dtheta = []
        Dtheta = np.append(Dtheta, self.dtheta)
        Fd = []
        Control = []
        Fd_gt = []
        Fd_hat = []
        A1 = [np.copy(self.a1)]
        # print('The initial wind is w_x = %.2f, w_y = %.2f' % (self.Wind_x, self.Wind_y))
        
        while True:
            self.process()
            Theta = np.append(Theta, self.theta)
            Dtheta = np.append(Dtheta, self.dtheta)
            Control = np.append(Control, self.u)
            Fd = np.append(Fd, self.F_d_data)
            Fd_gt = np.append(Fd_gt, self.F_d_gt)
            Fd_hat = np.append(Fd_hat, self.F_d_hat)
            
            # if not self.total_step % int(1.0 / self.step_size):
            # print('Simulation time: ' + str(self.total_step*self.step_size))

            if not self.total_step % int(2.0 / self.step_size):
                self.total_wind += 1
                self.Wind_x = self.Wind[self.total_wind][0]
                self.Wind_y = self.Wind[self.total_wind][1]    
                self.trigger = True
                self.sub_step = 0
                A1.append(np.copy(self.a1))
                if not self.total_wind % 10:
                    pass
                    # print('Wind is changed to w_x = %.2f, w_y = %.2f' % (self.Wind_x, self.Wind_y))
                
            if self.step_size*self.total_step >= self.duration:
                break

        Output = {'theta':Theta[:-1], 'dtheta':Dtheta[:-1], 'f_oracle':Fd_gt, \
                  'f_data':Fd, 'f_hat':Fd_hat, 'u':Control, 'a1':A1}
                
        error = np.mean(np.sqrt(Output['theta']**2 + Output['dtheta']**2))
        print('ACE: %.3f' % error)

        return Output, error

# Bilinear model, \hat{f}(x,c) = Y(x)A\alpha(c)
class Pendulum_B(Pendulum):

    def __init__(self, Wind, Noise, init_theta=0.0, init_dtheta=0.0, duration=20, Cd=0.3, \
                 alpha=0.5, mode='baseline', eta1_base=0.01, eta2_base=0.01):
        self.duration = duration
        
        # Pendulum parameters
        self.m = 1.0
        self.l = 0.5
        self.g = 9.81                               
        self.g_hat = 9.0
        
        # States                 
        self.theta = init_theta                   
        self.dtheta = init_dtheta                                                               
        self.state = np.array([self.theta, self.dtheta])
        self.prev_state = self.state
        self.dstate = np.array([0.0, 0.0])
        self.u = 0.0
        
        # Control gain
        self.gain = 1.5
        
        # Noise
        self.Noise = Noise
        self.a_noise = 0.0
        
        # Step and meta step
        self.step_size = 1e-2
        self.total_step = 0 
        self.total_wind = 0
        self.sub_step = 0
        self.trigger = False
    
        # Wind condition
        self.Wind = Wind
        self.Wind_x = self.Wind[self.total_wind][0]
        self.Wind_y = self.Wind[self.total_wind][1]
        self.Cd = Cd
        self.alpha = alpha
    
        # Fd gt and measurement
        self.F_d_data = 0.0
        self.F_d_gt = 0.0
        self.F_d_hat = 0.0
    
        # Learning method
        self.mode = mode
        
        # Kernel initialization 
        self.dim_A = 30
        self.dim_a = 20
        self.w = np.random.normal(size=(self.dim_A, 2))
        self.b = np.random.uniform(low=0, high=2*np.pi, size=self.dim_A)
        self.A = np.random.normal(size=(self.dim_A, self.dim_a))
        self.A /= np.linalg.norm(self.A, 2)
        self.a = np.zeros(self.dim_a)
        self.A_vector = np.zeros(self.dim_A)
        self.y = 0.0
        
        self.a_batch = [] # for meta-adaptation
        self.F_d_data_batch = [] # for meta-adaptation
        self.state_batch = [] # for meta-adaptation
    
        # Learning rate
        self.eta1 = 0.0
        self.eta2 = 0.0
        self.eta1_base = eta1_base
        self.eta2_base = eta2_base
    
    def kernel(self):
        if self.mode == 'meta-adapt':
            self.y = np.dot(np.cos(self.w @ self.state + self.b) @ self.A, self.a)
        elif self.mode == 'adapt':
            self.y = np.dot(np.cos(self.w @ self.state + self.b), self.A_vector)
        else:
            raise Exception('Not Implemented!')
        return self.y
    
    def adapt(self):
        self.eta1 = self.eta1_base / np.sqrt(self.sub_step)
        self.A_vector -= self.eta1 * (self.y-self.F_d_data) * np.cos(self.w @ self.prev_state + self.b)
        
    def meta_adapt(self):
        self.eta2 = self.eta2_base / np.sqrt(self.sub_step)
        self.a -= self.eta2 * (self.y-self.F_d_data) * np.cos(self.w @ self.prev_state + self.b) @ self.A
        
        self.a_batch.append(np.copy(self.a))
        self.F_d_data_batch.append(np.copy(self.F_d_data))
        self.state_batch.append(np.copy(self.prev_state))
        
        if self.trigger:
            self.eta1 = self.eta1_base / np.sqrt(self.total_wind)
            A_grad = np.zeros_like(self.A)
            for i in range(len(self.a_batch)):
                a = self.a_batch[i]
                f = self.F_d_data_batch[i]
                s = self.state_batch[i]
                Y = np.cos(self.w @ s + self.b)
                y = np.dot(Y @ self.A, a)
                A_grad += (y - f) * np.outer(Y, a)
            self.A -= self.eta1 * A_grad
            self.trigger = False
            self.a_batch = []
            self.F_d_data_batch = []
            self.state_batch = []
            
    def simulate(self):
        Theta = []
        Theta = np.append(Theta, self.theta)
        Dtheta = []
        Dtheta = np.append(Dtheta, self.dtheta)
        Fd = []
        Control = []
        Fd_gt = []
        Fd_hat = []
        A = []
        a = []
        # print('The initial wind is w_x = %.2f, w_y = %.2f' % (self.Wind_x, self.Wind_y))
        
        while True:
            self.process()
            Theta = np.append(Theta, self.theta)
            Dtheta = np.append(Dtheta, self.dtheta)
            Control = np.append(Control, self.u)
            Fd = np.append(Fd, self.F_d_data)
            Fd_gt = np.append(Fd_gt, self.F_d_gt)
            Fd_hat = np.append(Fd_hat, self.F_d_hat)
            a.append(np.copy(self.a))
            
            # if not self.total_step % int(1.0 / self.step_size):
            # print('Simulation time: ' + str(self.total_step*self.step_size))

            if not self.total_step % int(2.0 / self.step_size):
                self.total_wind += 1
                self.Wind_x = self.Wind[self.total_wind][0]
                self.Wind_y = self.Wind[self.total_wind][1] 
                self.trigger = True
                self.sub_step = 0
                A.append(np.copy(self.A))
                if not self.total_wind % 10:
                    pass
                    # print('Wind is changed to w_x = %.2f, w_y = %.2f' % (self.Wind_x, self.Wind_y))
                
            if self.step_size*self.total_step >= self.duration:
                break

        Output = {'theta':Theta[:-1], 'dtheta':Dtheta[:-1], 'f_oracle':Fd_gt, \
                  'f_data':Fd, 'f_hat':Fd_hat, 'u':Control, 'A':A, 'a':a}
        
        error = np.mean(np.sqrt(Output['theta']**2 + Output['dtheta']**2))
        print('ACE: %.3f' % error)

        return Output, error

class Phi(nn.Module):

    def __init__(self):
        super(Phi, self).__init__()
        self.fc1 = spectral_norm(nn.Linear(2, 25))
        self.fc2 = spectral_norm(nn.Linear(25, 30))
        self.fc3 = spectral_norm(nn.Linear(30, 20))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Deep OMAC, \hat{f}(x,c) = NN(x)\alpha(c)
class Pendulum_D(Pendulum_B):

    def __init__(self, Wind, Noise, init_theta=0.0, init_dtheta=0.0, duration=20, Cd=0.3, \
                 alpha=0.5, mode='baseline', eta1_base=0.01, eta2_base=0.01):
        self.duration = duration
        
        # Pendulum parameters
        self.m = 1.0
        self.l = 0.5
        self.g = 9.81                               
        self.g_hat = 9.0
        
        # States                 
        self.theta = init_theta                   
        self.dtheta = init_dtheta                                                               
        self.state = np.array([self.theta, self.dtheta])
        self.prev_state = self.state
        self.dstate = np.array([0.0, 0.0])
        self.u = 0.0
        
        # Control gain
        self.gain = 1.5
        
        # Noise
        self.Noise = Noise
        self.a_noise = 0.0
        
        # Step and meta step
        self.step_size = 1e-2
        self.total_step = 0 
        self.total_wind = 0
        self.sub_step = 0
        self.trigger = False
    
        # Wind condition
        self.Wind = Wind
        self.Wind_x = self.Wind[self.total_wind][0]
        self.Wind_y = self.Wind[self.total_wind][1]
        self.Cd = Cd
        self.alpha = alpha
    
        # Fd gt and measurement
        self.F_d_data = 0.0
        self.F_d_gt = 0.0
        self.F_d_hat = 0.0
    
        # Learning method
        self.mode = mode
        
        # Kernel initialization 
        self.dim_a = 20
        self.phi = Phi()
        self.a = np.zeros(self.dim_a)
        self.y = 0.0

        # For neural network training
        self.optimizer = optim.Adam(self.phi.parameters(), lr=eta1_base)
        self.loss = nn.MSELoss()
        self.a_batch = [] # for meta-adaptation
        self.F_d_data_batch = [] # for meta-adaptation
        self.state_batch = [] # for meta-adaptation
        self.Loss = 0.0

        # Learning rate
        self.eta1 = 0.0
        self.eta2 = 0.0
        self.eta1_base = eta1_base
        self.eta2_base = eta2_base
    
    def kernel(self):
        if self.mode == 'meta-adapt':
            temp = torch.from_numpy(self.state)
            with torch.no_grad():
                self.y = np.dot(self.phi(temp).numpy(), self.a)
        else:
            raise Exception('Not Implemented!')
        return self.y

    def meta_adapt(self):
        self.eta2 = self.eta2_base / np.sqrt(self.sub_step)
        temp = torch.from_numpy(self.prev_state)
        with torch.no_grad():
            self.a -= self.eta2 * (self.y-self.F_d_data) * self.phi(temp).numpy()
        
        self.a_batch.append(np.copy(self.a))
        self.F_d_data_batch.append(np.copy(self.F_d_data))
        self.state_batch.append(np.copy(self.prev_state))
        
        if self.trigger:
            self.optimizer.zero_grad()

            a_batch = torch.from_numpy(np.array(self.a_batch))
            F_d_data_batch = torch.from_numpy(np.array(self.F_d_data_batch))
            state_batch = torch.from_numpy(np.array(self.state_batch))
            phi_batch = self.phi(state_batch)
            loss = 0.
            for i in range(len(phi_batch)):
                temp = torch.dot(phi_batch[i,:], a_batch[i,:])
                loss += self.loss(temp, F_d_data_batch[i])

            loss.backward()
            self.optimizer.step()
            self.Loss = loss.item()

            self.trigger = False
            self.a_batch = []
            self.F_d_data_batch = []
            self.state_batch = []
            
    def simulate(self):
        Theta = []
        Theta = np.append(Theta, self.theta)
        Dtheta = []
        Dtheta = np.append(Dtheta, self.dtheta)
        Fd = []
        Control = []
        Fd_gt = []
        Fd_hat = []
        a = []
        Loss = []
        # print('The initial wind is w_x = %.2f, w_y = %.2f' % (self.Wind_x, self.Wind_y))
        
        while True:
            self.process()
            Theta = np.append(Theta, self.theta)
            Dtheta = np.append(Dtheta, self.dtheta)
            Control = np.append(Control, self.u)
            Fd = np.append(Fd, self.F_d_data)
            Fd_gt = np.append(Fd_gt, self.F_d_gt)
            Fd_hat = np.append(Fd_hat, self.F_d_hat)
            a.append(np.copy(self.a))
            
            # if not self.total_step % int(1.0 / self.step_size):
            # print('Simulation time: ' + str(self.total_step*self.step_size))

            if not self.total_step % int(2.0 / self.step_size):
                self.total_wind += 1
                self.Wind_x = self.Wind[self.total_wind][0]
                self.Wind_y = self.Wind[self.total_wind][1] 
                self.trigger = True
                self.sub_step = 0
                Loss.append(self.Loss)
                if not self.total_wind % 10:
                    pass
                    # print('Wind is changed to w_x = %.2f, w_y = %.2f' % (self.Wind_x, self.Wind_y))
                
            if self.step_size*self.total_step >= self.duration:
                break

        Output = {'theta':Theta[:-1], 'dtheta':Dtheta[:-1], 'f_oracle':Fd_gt, \
                  'f_data':Fd, 'f_hat':Fd_hat, 'u':Control, 'loss':Loss, 'a':a}
        
        error = np.mean(np.sqrt(Output['theta']**2 + Output['dtheta']**2))
        print('ACE: %.3f' % error)

        return Output, error

def plot(pendulum, Output, ax1, ax2, name, ylim1, ylim2, xlim, ylabel=False, legend=False):
    time = np.linspace(1e-2, pendulum.duration, int(pendulum.duration*1e2))

    ax1.plot(time, Output['theta'])
    ax1.plot(time, Output['dtheta'])
    if legend:
        ax1.legend([r'$\theta$', r'$\dot{\theta}$'], fontsize=15)
    ax1.axhline(y=0, linestyle='--', color='k')
    error = np.mean(np.sqrt(Output['theta']**2 + Output['dtheta']**2))
    ax1.set_title(name + '\n ACE: %.3f' % error, fontsize=15)
    ax1.set_ylim(ylim1)
    ax1.set_xlim(xlim)
    
    ax2.plot(time, Output['f_oracle'])
    ax2.plot(time, Output['f_hat'])
    if legend:
        ax2.legend([r'$f$', r'$\hat{f}$'], fontsize=15)
    ax2.set_xlabel('time (s)', fontsize=15)
    if ylabel:
        ax2.set_ylabel(r'$N\cdot m$', fontsize=15)
    ax2.set_ylim(ylim2)
    ax2.set_xlim(xlim)
    
    for i in range(int(pendulum.duration)):
        if not i % 2 and i > 0:
            ax1.axvline(x=i, linestyle='--', color='r', alpha=0.3)
            ax2.axvline(x=i, linestyle='--', color='r', alpha=0.3)

def plot_bound(Output):
    ax1_min = min(min(Output['theta']), min(Output['dtheta']))
    ax1_max = max(max(Output['theta']), max(Output['dtheta']))
    ax2_min = min(np.min(Output['f_oracle']), min(Output['f_hat']))
    ax2_max = max(np.max(Output['f_oracle']), max(Output['f_hat']))
    return [np.array([ax1_min, ax2_min]), np.array([ax1_max, ax2_max])]