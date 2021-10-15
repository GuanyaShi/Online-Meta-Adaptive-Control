''' You should add a docstring here '''

import copy
from warnings import warn
import random
from collections import namedtuple
import pkg_resources

import numpy as np
from torch import log
from tqdm import tqdm
import rowan  # NOTE: this is slow, but easier to use than numpy quaternion tools (np ~4x faster than rowan)

# from . import controller as controllers
from .trajectory import fig8
from .utils import readparamfile

__author__ = "Michael O'Connell"
__date__ = "Octoboer 2021"

# Data = namedtuple('Data', 'log metadata')

DEFAULT_QUAD_PARAMETER_FILE = pkg_resources.resource_filename(__name__, 'params/quadrotor.json')

class Quadrotor():
    def __init__(self, paramsfile=DEFAULT_QUAD_PARAMETER_FILE, **kwargs):
        self.params = readparamfile(paramsfile)
        self.params.update(kwargs)
        self.B = np.array([self.params['C_T'] * np.ones(4), 
                           self.params['C_T'] * self.params['l_arm'] * np.array([-1., -1., 1., 1.]),
                           self.params['C_T'] * self.params['l_arm'] * np.array([-1., 1., 1., -1.]),
                           self.params['C_q'] * np.array([-1., 1., -1., 1.])])
        self.params['J'] = np.array(self.params['J'])
        self.params['process_noise_covariance'] = np.array(self.params['process_noise_covariance'])
        self.params['imu_covariance'] = np.array(self.params['imu_covariance'])
        self.Jinv = np.linalg.inv(self.params['J'])
        l_arm = self.params['l_arm'] # Notice l_arm is the x/y component of the offset
        h = self.params['h']
        D = self.params['D']
        self.stick_figure = {'lines': (((0.,0.,0.),(l_arm,l_arm,0.)),
                                      ((0.,0.,0.),(-l_arm,l_arm,0.)),
                                      ((0.,0.,0.),(-l_arm,-l_arm,0.)),
                                      ((0.,0.,0.),(l_arm,-l_arm,0.)),
                                      ((l_arm,l_arm,0.),(l_arm,l_arm,h)),
                                      ((-l_arm,l_arm,0.),(-l_arm,l_arm,h)),
                                      ((-l_arm,-l_arm,0.),(-l_arm,-l_arm,h)),
                                      ((l_arm,-l_arm,0.),(l_arm,-l_arm,h)),
                                      ),
                            'circles': (((l_arm, l_arm, h), D/2, (0., 0., l_arm)),
                                        ((-l_arm, l_arm, h), D/2, (0., 0., l_arm)),
                                        ((-l_arm, -l_arm, h), D/2, (0., 0., l_arm)),
                                        ((l_arm, -l_arm, h), D/2, (0., 0., l_arm)),)
        }
        self.t_last_wind_update = 0.

    def update_motor_speed(self, u, dt, Z=None):
        # Hidden motor dynamics discrete update
        if Z is None:
            Z = u
        else:
            alpha_m = 1 - np.exp(-self.params['w_m'] * dt)
            Z = alpha_m*u + (1-alpha_m)*Z
        
        return np.maximum(np.minimum(Z, self.params['motor_max_speed']), self.params['motor_min_speed'])

    def f(self, X, Z, _t, logentry=None):
        # x = X[0]
        # y = X[1]
        # z = X[2]
        # q = X[3:7]
        # vx = X[7]
        # vy = X[8]
        # vz = X[9]
        # wx = X[10]
        # wy = X[11]
        # wz = X[12]
        p = X[0:3]
        q = X[3:7]
        R = rowan.to_matrix(q)
        v = X[7:10]
        w = X[10:]
        # th = X[2] 
        # xdot = X[3]
        # ydot = X[4]
        # thdot = X[5]

        # Thruster force model
        T, *tau = self.B @ (Z ** 2)
        
        # Position kinematics
        Xdot = np.empty(13)
        Xdot[0:3] = v

        # Quaternion attitude kinematics
        Xdot[3:7] = rowan.calculus.derivative(q, w) # NOTE: this is slow........
        
        # Acceleration model
        F = np.empty(3)
        # T = self.params['C_T'] * sum(Z ** 2)
        F = T * (R @ np.array([0., 0., 1.])) - np.array([0., 0., self.params['g']*self.params['m']])
        a = F / self.params['m']
        Xdot[7:10] = a
        
        # Angular accelartion model
        alpha = np.linalg.solve(self.params['J'], np.cross(self.params['J'] @ w, w) + tau)
        # alpha = np.linalg.solve(self.params['J'], tau)
        Xdot[10:] = alpha

        # if logentry is not None:
        logentry['T'] = T
        logentry['tau'] = tau
        logentry['alpha_'] = alpha
        logentry['crossterm'] = np.cross(self.params['J'] @ w, w)
        # logentry['z_body'] = R @ np.array((0., 0., 1.))
        
        return Xdot

    _step_output = namedtuple('_step_output', 'X t Z Xdot')

    def step(self, X, u, t, dt, Z=None, logentry=None):
        if self.params['integration_method'] == 'rk4':
            # RK4 method
            Z = self.update_motor_speed(Z=Z, u=u, dt=0.0)
            k1 = dt * self.f(X, Z, t, logentry=logentry)
            Z = self.update_motor_speed(Z=Z, u=u, dt=dt/2)
            k2 = dt * self.f(X + k1/2, Z, t+dt/2)
            k3 = dt * self.f(X + k2/2, Z, t+dt/2)
            Z = self.update_motor_speed(Z=Z, u=u, dt=dt/2)
            k4 = dt * self.f(X + k3, Z, t+dt)
            
            Xdot = (k1 + 2*k2 + 2*k3 + k4)/6 / dt
            X = X + (k1 + 2*k2 + 2*k3 + k4)/6
        elif self.params['integration_method'] == 'euler':
            Xdot = self.f(X,Z,t, logentry=logentry)
            X = X + dt * Xdot
            Z = self.update_motor_speed(Z=Z, u=u, dt=dt)
        else:
            raise NotImplementedError
        # print('before normalization: ', X[3:7])
        X[3:7] = rowan.normalize(X[3:7])
        # print('after normalization: ', X[3:7])

        # Zero mean, Gaussian noise model
        noise = np.random.multivariate_normal(np.zeros(6), self.params['process_noise_covariance'])
        X[7:] += noise
        
        logentry['process_noise'] = noise
            
        return self._step_output(X, t+dt, Z, Xdot)

    def measurement(self, X, Z, t, Xdot, logentry=None):
        X_meas = X
        noise = np.random.multivariate_normal(np.zeros(6), self.params['process_noise_covariance'])
        imu_meas = Xdot[7:] + noise

        logentry['measurement_noise'] = noise

        return X_meas, imu_meas


    def runiter(self, trajectory, controller, X0=None, Z0=None):
        X = np.zeros(13)
        X[3] = 1.
        if Z0 is None:
            hover_motor_speed = np.sqrt(self.params['m'] * self.params['g'] / (4 * self.params['C_T']))
            Z = hover_motor_speed * np.ones(4)
        else:
            Z = Z0
        
        t = self.params['t_start'] # simulation time
        t_posctrl = -0.0 # time of next position control update
        t_attctrl = -0.0 # time of next attitude control update
        t_angratectrl = -0.0 # time of next rate control update
        t_readout = -0.0
        logentry = {}

        X_meas = X
        imu_meas = np.zeros(6)
        
        while t < self.params['t_stop']:

            if t >= t_posctrl:
                pd, vd, ad, jd, sd = trajectory(t)
                T_sp, q_sp = controller.position(X=X_meas, imu=imu_meas, pd=pd,
                                                 vd=vd, ad=ad, jd=jd, sd=sd, t=t,
                                                 logentry=logentry,
                                                 t_last_wind_update=self.t_last_wind_update)
                t_posctrl += controller.params['dt_posctrl']
            if t >= t_attctrl:
                w_sp = controller.attitude(q=X[3:7], q_sp=q_sp)
                t_attctrl += controller.params['dt_attctrl']
            if t >= t_angratectrl:
                torque_sp = controller.angrate(w=X[10:], w_sp=w_sp, 
                                               dt=controller.params['dt_angratectrl'],
                                               logentry=logentry)
                u = controller.mixer(torque_sp=torque_sp, T_sp=T_sp, logentry=logentry)
                t_angratectrl += controller.params['dt_angratectrl']
            step_output = self.step(X=X, u=u, t=t, dt=self.params['dt'], Z=Z, logentry=logentry)
            X = step_output.X
            t = step_output.t
            Z = step_output.Z
            Xdot = step_output.Xdot
            X_meas, imu_meas = self.measurement(X=X, Z=Z, t=t, Xdot=Xdot, logentry=logentry)

            if t >= t_readout:
                logentry['t'] = t
                logentry['X'] = X
                logentry['Xdot'] = Xdot
                logentry['Z'] = Z
                logentry['T_r'] = T_sp
                logentry['q_sp'] = q_sp
                # logentry['w_d'] = w_d
                # logentry['alpha_d'] = alpha_d
                logentry['pd'] = pd
                logentry['vd'] = vd
                logentry['ad'] = ad
                # logentry['jd'] = jd
                # logentry['sd'] = sd
                yield copy.deepcopy(logentry)
                logentry['meta_adapt_trigger'] = False
                t_readout += self.params['dt_readout']

    def run(self, controller, trajectory=fig8, show_progress=True, seed=None):
        if type(seed) is not None:
            random.seed(seed)
            np.random.seed(seed)
        controller.reset_controller()
        # # Use zip to switch output array from indexing of time, (X, t, log), to indexing of (X, t, log), time
        # log = zip(*self.runiter(trajectory=trajectory, controller=controller))
        if show_progress:
            log = list(tqdm(self.runiter(trajectory=trajectory, controller=controller), total=(self.params['t_stop']-self.params['t_start'])/self.params['dt_readout']))
        else:
            log = list(self.runiter(trajectory=trajectory, controller=controller))
        # Concatenate entries of the log dictionary into single np.array, with first dimension corresponding to each time step recorded
        log2 = {k: np.array([logentry[k] for logentry in log]) for k in log[0]}
        return log2
        # metadata = {}
        # metadata.update(self.metadata)
        # metadata.quad_params = {}
        # metadata.quad_params.update(self.params)
        # metadata.update(trajectory.metadata) # TODO: rewrite trajectories as classes with a metadata attribute
        # return Data(log2, metadata)

class QuadrotorWithSideForce(Quadrotor):
    def __init__(self, *args, sideforcemodel='force and torque', Vwind=0., Vwind_cov=0., Vwind_gust=0., wind_constraint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sideforcemodel = sideforcemodel
        self.t_wind_update = np.nan

        Vwind = np.array(Vwind)
        if Vwind.shape == (): # Vwind is scalar
            Vwind = Vwind * np.array((1., 0., 0.))
        elif Vwind.shape == (3,):
            pass
        else:
            raise ValueError("Vwind should be a 3-vector or scalar")

        self.Vwind = Vwind
        self.Vwind_mean = Vwind.copy()

        Vwind_cov = np.array(Vwind_cov)
        if Vwind_cov.shape == (): # Vwind is scalar
            Vwind_cov = Vwind_cov * np.diag((1., 0., 0.))
        elif Vwind_cov.shape == (3,):
            Vwind_cov = np.diag(Vwind_cov)
        elif Vwind_cov.shape == (3,3):
            pass
        else:
            raise ValueError("Vwind_cov should be a 3x3 matrix, 3-vector or scalar")

        self.Vwind_cov = Vwind_cov

        assert(type(Vwind_gust) == float)
        if wind_constraint == 'soft':
            # Vwind_gust is sqrt(E[|Vwind - Vwind0|^2])
            Vwind_damping = np.trace(Vwind_cov) / (Vwind_gust/1.25)**2 # divide gust by value tuned to give similar mean wind difference for soft and hard constraints
            if Vwind_damping * self.params['dt'] > 1: # for convergence of discrete wind speed update, 0 < Vwind_damping * dt < 2
                warn('Vwind_cov too high, Vwind_gust not used')
                Vwind_damping = 1 / self.params['dt']
            self.Vwind_damping = Vwind_damping
        elif wind_constraint == 'hard':
            self.Vwind_gust = Vwind_gust
        elif wind_constraint is None:
            pass
        else:
            raise ValueError("Unknown wind speed constraint")
        self.wind_constraint = wind_constraint

        l_arm = self.params['l_arm']
        h = self.params['h']
        self.r_arms = np.array(((l_arm, -l_arm, h), 
                                (-l_arm, -l_arm, h),
                                (-l_arm, l_arm, h),
                                (l_arm, l_arm, h)))

        # self.metadata['dynamics_model'] = 'wind side-force, ' + sideforcemodel 

        self.t_last_wind_update = 0.

    def get_wind_velocity(self, t):
        dt = (t - self.t_last_wind_update)
        if dt > self.params['wind_update_period']:
            self.Vwind += np.random.multivariate_normal(mean = np.zeros(3), cov = (self.Vwind_cov * dt))

            if self.wind_constraint is None:
                pass

            elif self.wind_constraint == 'soft':
                self.Vwind -= self.Vwind_damping * (self.Vwind - self.Vwind_mean) * dt

            elif self.wind_constraint == 'hard':
                Vwind_diff = self.Vwind - self.Vwind_mean
                if np.linalg.norm(Vwind_diff) > self.Vwind_gust:
                    self.Vwind = self.Vwind_mean +  Vwind_diff * self.Vwind_gust / np.linalg.norm(Vwind_diff)

            self.t_last_wind_update = t

        return self.Vwind # + 2 * np.array((np.sign(np.sin(1/3 * t)), 0., 0.))
        # if t != self.t_wind_update:
        #     self.wind_velocity = self.params.Vwind_mean + np.dot(self.params.Vwind_cov, np.random.normal(size=2))
        #     # TODO: implement wind as band limited Gaussian process
        # return self.wind_velocity

    def f(self, X, Z, t, logentry=None):
        Xdot = super().f(X,Z,t,logentry)
        
        q = X[3:7]
        v = X[7:10]
        R_world_to_body = rowan.to_matrix(q).transpose() # NOTE: this is slow........

        # Side force model
        # R_world_to_body = np.array([[np.cos(th), np.sin(th)],[-np.sin(th), np.cos(th)]])
        Vwind = self.get_wind_velocity(t) 			 # velocity of wind in world frame
        Vinf = Vwind - v		                     # velocity of air relative to quadrotor in world frame orientation
        Vinf_B = R_world_to_body @ Vinf              # use _B suffix to denote vector is in body frame
        Vz_B = np.array((0., 0., Vinf_B[2]))         # z-body-axes aligned relative wind velocity
        Vs_B = Vinf_B - Vz_B		                 # cross wind
        if np.linalg.norm(Vs_B) > 1e-4:
            aoa = np.arcsin(np.linalg.norm(Vz_B)/np.linalg.norm(Vinf_B)) # angle of attack
            n = np.sqrt(Z / self.params['C_T'])            # Propeller rotation speed
            Fs_per_prop = self.params['C_s'] * self.params['rho'] * (n ** self.params['k1']) \
                   * (np.linalg.norm(Vinf) ** (2 - self.params['k1'])) * (self.params['D'] ** (2 + self.params['k1'])) \
                   * ((np.pi / 2) ** 2 - aoa ** 2) * (aoa + self.params['k2'])
            Fs_B = (Vs_B/np.linalg.norm(Vs_B)) * sum(Fs_per_prop)
            # print(aoa, n, self.params.g, F, Fs_B)
            
            tau_s = np.zeros(3)
            for i in range(4):
                Fs_B_singleprop = (Vs_B/np.linalg.norm(Vs_B)) * Fs_per_prop[i]
                tau_s += np.cross(self.r_arms[i], Fs_B_singleprop)
        else:
            Fs_per_prop = np.zeros((4,3))
            Fs_B = np.zeros(3) # For case when relative wind speed is close to zero
            tau_s = np.zeros(3)
            
        Fs = R_world_to_body.transpose() @ Fs_B
        # tau_s = - Fs_B[0] * (self.params.D / 4) 
        
        if self.sideforcemodel == 'force and torque':
            pass
        elif self.sideforcemodel == 'force only':
            tau_s = 0.*tau_s
        elif self.sideforcemodel == 'torque only':
            Fs = 0.*Fs
        elif self.sideforcemodel == 'none':
            tau_s *= 0.
            Fs *= 0.

        Xdot[7:10] += Fs / self.params['m']
        Xdot[10:] +=  self.Jinv @ tau_s

        # if logentry is not None:
        logentry['Fs'] = Fs
        # logentry['Fs_B'] = Fs_B
        logentry['tau_s'] = tau_s
        logentry['Vwind'] = Vwind

        return Xdot