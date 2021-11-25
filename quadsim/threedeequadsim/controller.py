# pylint: disable=invalid-name

from warnings import warn
import collections
import pkg_resources

import numpy as np
import rowan

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
torch.set_default_tensor_type('torch.DoubleTensor')

from .utils import readparamfile


__author__ = "Michael O'Connell"
__date__ = "Octoboer 2021"
__copyright__ = "Copyright 2021 by Michael O'Connell"
__credits__ = ["Michael O'Connell", "Guanya Shi", "Kamyar Azizzadenesheli", "Soon-Jo Chung", "Yisong Yue"]
__maintainer__ = "Michael O'Connell"
__email__ = "moc@caltech.edu"
__status__ = "Prototype"


DEFAULT_CONTROL_PARAM_FILE = pkg_resources.resource_filename(__name__, 'params/controller.json')
DEFAULT_PX4_PARAM_FILE = pkg_resources.resource_filename(__name__, 'params/px4.json')
DEFAULT_QUAD_PARAMETER_FILE = pkg_resources.resource_filename(__name__, 'params/quadrotor.json')

# Define a named tuple for force so that subclasses can also pass around derivatives of force
Force = collections.namedtuple('Force', 'F F_dot F_ddot', defaults=(np.zeros(3), np.zeros(3)))


def ignore(*args, **kwargs):
    pass

class Controller():
    ''' Controller class implements the attitude controller and thrust mixing. The position controller is not implemented and should be implemented in child classes.'''
    _name = None
    name_long = None
    def __init__(self, quadparamfile=DEFAULT_QUAD_PARAMETER_FILE, px4paramfile=DEFAULT_PX4_PARAM_FILE):
        self.params = readparamfile(quadparamfile)

        self.px4_params = readparamfile(px4paramfile) 
        self.px4_params['angrate_max'] = np.array((self.px4_params['MC_ROLLRATE_MAX'],
                                                  self.px4_params['MC_PITCHRATE_MAX'],
                                                  self.px4_params['MC_YAWRATE_MAX']))
        self.px4_params['angrate_gain_P'] = np.diag((self.px4_params['MC_ROLLRATE_P'],
                                                  self.px4_params['MC_PITCHRATE_P'],
                                                  self.px4_params['MC_YAWRATE_P']))
        self.px4_params['angrate_gain_I'] = np.diag((self.px4_params['MC_ROLLRATE_I'],
                                                  self.px4_params['MC_PITCHRATE_I'],
                                                  self.px4_params['MC_YAWRATE_I']))
        self.px4_params['angrate_gain_D'] = np.diag((self.px4_params['MC_ROLLRATE_D'],
                                                  self.px4_params['MC_PITCHRATE_D'],
                                                  self.px4_params['MC_YAWRATE_D']))
        self.px4_params['angrate_gain_K'] = np.diag((self.px4_params['MC_ROLLRATE_K'],
                                                  self.px4_params['MC_PITCHRATE_K'],
                                                  self.px4_params['MC_YAWRATE_K']))
        self.px4_params['angrate_int_lim'] = np.array((self.px4_params['MC_RR_INT_LIM'],
                                                   self.px4_params['MC_PR_INT_LIM'],
                                                   self.px4_params['MC_YR_INT_LIM']))
        self.px4_params['attitude_gain_P'] = np.diag((self.px4_params['MC_ROLL_P'],
                                                  self.px4_params['MC_PITCH_P'],
                                                  self.px4_params['MC_YAW_P']))
        # self.px4_params.attitude_max = np.array((self.px4_params.MC_ROLL_MAX,
        #                                           self.px4_params.MC_PITCH_MAX,
        #                                           self.px4_params.MC_YAW_MAX))
        self.px4_params['angacc_max'] = np.array(self.px4_params['angacc_max'])
        self.px4_params['J'] = np.array(self.px4_params['J'])
        self.B = None
    
    def position(self, X, imu=np.zeros(6), pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), sd=np.zeros(3), logentry=None):
        raise NotImplementedError

    def reset_controller(self):

        # Reset Angular rate control parameters
        self.w_error_integral = np.zeros(3)
        self.w_filtered = np.zeros(3)
        self.w_filtered_last = np.zeros(3)

    def limit(self, array, upper_limit, lower_limit=None):
        """ Ensure upper_limit >= array >= lower_limit,
        array must be mutable and is modified in place """
        if lower_limit is None:
            lower_limit = - upper_limit
        array[array > upper_limit] = upper_limit[array > upper_limit]
        array[array < lower_limit] = lower_limit[array < lower_limit]


    def attitude(self, q, q_sp, w_d=np.zeros(3)):
        """
        MC Attitude Controller for PX4 
        https://dev.px4.io/master/en/flight_stack/controller_diagrams.html
        """
        q_error = rowan.multiply(rowan.inverse(q), q_sp)
        omega_sp = 2 * self.px4_params['attitude_gain_P'] @ (np.sign(q_error[0]) * q_error[1:])
        # omega_sp += w_d
        self.limit(omega_sp, self.px4_params['angrate_max'])
        return omega_sp

    def angrate(self, w, w_sp, dt, logentry):
        """
        Angular rate controller that matches PX4 implementation here:
        https://docs.px4.io/master/en/config_mc/pid_tuning_guide_multicopter.html#rate-controller

        """
        # Calculate the angular rate error
        w_error = w_sp - w

        # Integrate the error signal and limit it
        self.w_error_integral += dt * w_error
        self.limit(self.w_error_integral, self.px4_params['angrate_int_lim'])
        # Sanity check the limit function since you never bothered to write a unit test for it
        if any(self.w_error_integral > self.px4_params['angrate_int_lim']) or \
                any(self.w_error_integral < -self.px4_params['angrate_int_lim']) :
            raise ValueError

        # Calculate the derivative of the filtered angular rate
        # PX4 does not include derivative of setpoint
        const_w_filter = np.exp(- dt / self.px4_params['w_filter_time_const'])
        self.w_filtered *= const_w_filter
        self.w_filtered += (1 - const_w_filter) * w
        
        w_filtered_derivative = (self.w_filtered - self.w_filtered_last) / dt
        logentry['w_filtered_last'] = self.w_filtered_last.copy()
        self.w_filtered_last[:] = self.w_filtered[:] # Python is a garbage language

        alpha_sp = self.px4_params['angrate_gain_K'] \
                    @ (self.px4_params['angrate_gain_P'] @ w_error 
                       + self.px4_params['angrate_gain_I'] @ self.w_error_integral
                       - self.px4_params['angrate_gain_D'] @ w_filtered_derivative)

        self.limit(alpha_sp, self.px4_params['angacc_max'])
        # torque_sp = self.px4_params['J'] @ alpha_sp
        torque_sp = alpha_sp

        logentry['w'] = w
        logentry['w_error_integral'] = self.w_error_integral
        logentry['w_filtered'] = self.w_filtered
        logentry['w_filtered_derivative'] = w_filtered_derivative
        logentry['alpha_sp'] = alpha_sp

        return torque_sp

    def mixer_get_motorspeedsquared(self, torque_sp, T_sp):
        return np.linalg.solve(self.B, np.concatenate(((T_sp,), torque_sp)))

    def mixer(self, torque_sp, T_sp, logentry):
        """ Calculate motor speed commands from torque and thrust set points 

            - Assumes T_sp > 0
            - ~~reduces torque_sp[2] first if max and min motor speeds violated~~
                ^ incomplete and commented out
        """
        omega_squared = np.linalg.solve(self.B, np.concatenate(((T_sp,), torque_sp)))
        omega = np.sqrt(np.maximum(omega_squared, self.params['motor_min_speed']))
        omega = np.minimum(omega, self.params['motor_max_speed'])

        logentry['motor_speed_command'] = omega

        return omega

    def calculate_derivative(self, x, x_last, dt_inv):
        """
        Calculate the x and handles the case where x_last or dt_inv are initialized to None
        """ 
        try:
            x_dot = (x - x_last) * dt_inv
            x_last = x.copy()
        except TypeError as err:
            if x_last is None or dt_inv is None:
                x_dot = np.zeros_like(x)
                x_last = x.copy()
                return x_dot, x_last
            else:
                raise err
        return x_dot, x_last


class Baseline(Controller):
    ''' Baseline controller class with PID feedback and acceleration feedforward term and with exact quadrotor dynamic model'''
    _name = 'pid'
    name_long = 'PID'
    def __init__(self, ctrlparamfile=DEFAULT_CONTROL_PARAM_FILE, quadparamfile=DEFAULT_QUAD_PARAMETER_FILE, integral_control=True, **kwargs):
        super().__init__(quadparamfile=quadparamfile, **kwargs)
        self.params = readparamfile(filename=ctrlparamfile, params=self.params)
        self.integral_control = integral_control

    def calculate_gains(self):
        self.params['K_i'] = np.array(self.params['K_i'])

        if not self.integral_control:
            self.params['K_i'] = np.zeros((3, 3))
        self.params['K_p'] = np.diag([self.params['Lam_xy']*self.params['K_xy'],
                       self.params['Lam_xy']*self.params['K_xy'],
                       self.params['Lam_z']*self.params['K_z']])
        self.params['K_d'] = np.diag([self.params['K_xy'], self.params['K_xy'], self.params['K_z']])
        self.B = np.array([self.params['C_T'] * np.ones(4), 
                           self.params['C_T'] * self.params['l_arm'] * np.array([-1., -1., 1., 1.]),
                           self.params['C_T'] * self.params['l_arm'] * np.array([-1., 1., 1., -1.]),
                           self.params['C_q'] * np.array([-1., 1., -1., 1.])])
        # self.Binv = np.linalg.inv(self.B)


    def reset_controller(self):
        super().reset_controller()
        self.calculate_gains()
        self.F_r_dot = None
        self.F_r_last = None
        self.t_last = None
        self.t_last_wind_update = None
        self.p_error = np.zeros(3)
        self.v_error = np.zeros(3)
        self.int_error = np.zeros(3)
        self.dt = 0.
        self.dt_inv = 0.

    def get_q(self, F_r, yaw_desired, max_angle=np.pi, check=False):
        """ Finds quaterion that will give rotation to align body z-axis with with F_r and yield
        with desired yaw.
        """
        q_world_to_yawed = rowan.from_euler(0., 0., yaw_desired, 'xyz')

        rotation_axis = np.cross((0, 0, 1), F_r)
        if np.allclose(rotation_axis, (0., 0., 0.)):
            unit_rotation_axis = np.array((1., 0., 0.,))
        else:
            unit_rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_axis /= np.linalg.norm(F_r)
        rotation_angle = np.arcsin(np.linalg.norm(rotation_axis))
        if F_r[2] < 0:
            rotation_angle = np.pi - rotation_angle
        if rotation_angle > max_angle:
            rotation_angle = max_angle
        q_yawed_to_body = rowan.from_axis_angle(unit_rotation_axis, rotation_angle)

        q_r = rowan.multiply(q_world_to_yawed, q_yawed_to_body)

        if any(np.isnan(q_r)):
            raise TypeError
        
        return q_r

    def get_Fr(self, X, imu=np.zeros(3), pd=np.zeros(2), vd=np.zeros(2), ad=np.zeros(3), logentry=None, **kwargs):
        ignore(imu, logentry)

        self.p_error = p_error = - pd + X[0:3]
        self.v_error = v_error = - vd + X[7:10]
        self.int_error += self.dt * p_error

        a_r = - self.params['K_p'] @ p_error - self.params['K_d'] @ v_error \
              - self.params['K_i'] @ self.int_error + ad

        F_r = (a_r * self.params['m']) + np.array([0., 0., self.params['m'] * self.params['g']])

        try:
            lam = np.exp(- self.dt / self.params['force_filter_time_const'])
            # lam = 0.
            self.F_r_dot *= lam
            self.F_r_dot += (1 - lam) * (F_r - self.F_r_last) * self.dt_inv
        except TypeError as err:
            if self.F_r_last is None:
                self.F_r_dot = np.zeros(3)
            else:
                raise err

        if any(np.isinf(self.F_r_dot)):
            raise ValueError
        self.F_r_last = F_r.copy()

        logentry['p_error'] = p_error
        logentry['v_error'] = v_error
        logentry['p_term'] = - self.params['K_p'] @ p_error * self.params['m']
        logentry['v_term'] = - self.params['K_d'] @ v_error * self.params['m']
        logentry['i_term'] = - self.params['K_i'] @ self.int_error * self.params['m']
        logentry['ad_term'] = ad * self.params['m']
        logentry['g_term'] = np.array([0., 0., self.params['m'] * self.params['g']])
        
        return Force(F_r, self.F_r_dot)

    def position(self, X, *, t_last_wind_update, imu=np.zeros(6), pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), sd=np.zeros(3), t=None, logentry=None):
        try:
            self.dt = t - self.t_last
            self.dt_inv = 1 / self.dt
        except TypeError as err:
            if self.t_last is None or t is None:
                pass
            else:
                raise err

        try:
            if self.t_last_wind_update < t_last_wind_update:
                self.t_last_wind_update = t_last_wind_update
                meta_adapt_trigger = True
            else:
                meta_adapt_trigger = False
        except TypeError as err:
            self.t_last_wind_update = t_last_wind_update
            meta_adapt_trigger = False

        self.t_last = t
        force = self.get_Fr(X, imu=imu, pd=pd, vd=vd, ad=ad, meta_adapt_trigger=meta_adapt_trigger, logentry=logentry)
        F_r = force.F # *_r is the force command

        yaw_d = 0.

        # Compute thrust and quaternion with first order delay compensation
        F_r_dot = force.F_dot
        T_r_prime = np.linalg.norm(F_r + self.params['thrust_delay'] * F_r_dot)
        q_r_prime = self.get_q(F_r + self.params['attitude_delay'] * F_r_dot, yaw_d)
        F_r_prime = rowan.to_matrix(q_r_prime) @ np.array((0, 0, T_r_prime))

        # The following two blocks of code limit the commanded force to be within the maximum thrust and maximum attitude limits
        # Find projection of thrust onto cone within max_zenith_angle of zenith
        T_max = self.params['max_thrust']
        T_hover = self.params['m'] * self.params['g']
        if np.isnan(self.params['max_zenith_angle']):
            s_oncone = float('inf')
        elif self.params['max_zenith_angle'] >= np.pi - 1e-2:
            s_oncone = float('inf')
        elif np.sqrt(F_r_prime[0]**2 + F_r_prime[1]**2) < 1e-4:
            s_oncone = float('inf')
        elif F_r_prime[2] / np.sqrt(F_r_prime[0]**2 + F_r_prime[1]**2) >= (np.cos(self.params['max_zenith_angle'])/np.sin(self.params['max_zenith_angle'])):
            s_oncone = float('inf')
        else:
            s_oncone = T_hover / ((np.cos(self.params['max_zenith_angle'])/np.sin(self.params['max_zenith_angle'])) * np.sqrt(F_r_prime[0]**2 + F_r_prime[1]**2) - (F_r_prime[2] - T_hover))

        # Find projection of thrust onto sphere of max_thrust
        a = np.linalg.norm(F_r_prime - np.array([0., 0., T_hover])) ** 2
        b = 2 * T_hover * (F_r_prime[2] - T_hover)
        c = T_hover ** 2 - T_max ** 2
        if a >= 1e-4:
            s_onsphere = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        else:
            s_onsphere = float('inf')

        s = min(1, s_onsphere, s_oncone)
        F_r_prime = s * (F_r_prime - np.array([0., 0., T_hover])) + np.array([0., 0., T_hover])
        
        T_r_prime = np.linalg.norm(F_r_prime)
        q_r_prime = self.get_q(F_r_prime, yaw_d)

        if np.isnan(q_r_prime[0]):
            raise ValueError('quaternion was nan, maybe F_r[2] was 0 and the drone is trying to flip??')
        
        if T_r_prime > 124.:
            warn('thrust gets too high')
        T_d = self.params['m'] * np.linalg.norm(ad)
        q_d = self.get_q(ad + np.array((0., 0., self.params['g'])), yaw_d, check=True)
        # q_d_prime = self.get_q(ad - np.array((0., 0., self.params.g)) + self.params.attitude_delay * jd, yaw_d)
        # q_d = np.array((1., 0, 0, 0))

        if logentry is not None:
            logentry['q_d'] = q_d
            # logentry['q_d_prime'] = q_d_prime
            logentry['T_d'] = T_d
            logentry['F_r'] = F_r
            logentry['F_r_dot'] = F_r_dot
            logentry['F_r_prime'] = F_r_prime
            logentry['s_oncone'] = s_oncone
            logentry['s_onsphere'] = s_onsphere
            # logentry['int_error'] = self.int_error
            logentry['meta_adapt_trigger'] = meta_adapt_trigger

        return T_r_prime, q_r_prime


class MetaAdapt(Baseline):
    _name = None
    name_long = 'Meta Adaptive Control Template'
    # def __init__(self, adaptivectrlparamfile=DEFAULT_ADAPTIVE_PARAM_FILE, integral_control=False, *args, **kwargs):
        # self.params = readparamfile(filename=adaptivectrlparamfile, params=self.params)
    def __init__(self, integral_control=False, *args, **kwargs):
        super().__init__(integral_control=integral_control, *args, **kwargs)

        # Values to be initialized in reset_controller()
        self.motor_speed_command = None

    def calculate_gains(self):
        super().calculate_gains()
        self.Lambda = np.linalg.solve(self.params['K_d'], self.params['K_p'])

    def reset_controller(self):
        ret = super().reset_controller()
        if ret is not None:
            raise ValueError
        self.motor_speed_command = np.zeros(4)

    def get_residual(self, X, imu, logentry):
        ''' Compute a measurement of the residual force '''
        q = X[3:7]
        R = rowan.to_matrix(q) # body to world

        H = self.params['m'] * np.eye(3)
        C = np.zeros((3,3))
        g = np.array((0., 0., self.params['g'] * self.params['m']))
        T = self.params['C_T'] * sum(self.motor_speed_command ** 2)
        # T_alt, *_ = self.B @ (self.motor_speed_command ** 2)
        u = T * R @ np.array((0., 0., 1.))
        # f = - H @ imu[0:3] - C @ X[7:10] - g + u
        y = (H @ imu[0:3] + g - u)

        logentry['y'] = y
        logentry['mpddot'] = H @ imu[0:3]
        logentry['g'] = g
        logentry['u'] = - u

        return y

    def get_Fr(self, X, *, meta_adapt_trigger, imu=np.zeros(6), pd=np.zeros(2), vd=np.zeros(2), ad=np.zeros(3), logentry=None, **kwargs):
        ''' Override baseline force controller to also account for the adapted force '''
        # Get prior measurements. Note these can be the filtered measurements!
        y = self.get_residual(X, imu, logentry=logentry)
        fhat = self.get_f_hat(X)

        # Update parameters
        self.inner_adapt(X, fhat.F, y)

        self.update_batch(X, fhat.F, y)
        if meta_adapt_trigger:
            self.meta_adapt()

        # Get baseline controller force (without integral control)
        Fr = super().get_Fr(X, imu, pd, vd, ad, logentry)
        F_r, F_r_dot, F_r_ddot = Fr

        # Add in adaptive term
        f_hat = self.get_f_hat(X)

        F_r -= f_hat.F
        # F_r_dot -= f_hat.F_dot
        # F_r_ddot -= f_hat.F_ddot

        logentry['f_hat'] = f_hat
        logentry['F_r_no_adaptive'] = F_r.copy()
        logentry['pred_error'] = y - f_hat

        return Force(F_r, F_r_dot, F_r_ddot)

    def mixer(self, torque_sp, T_sp, logentry):
        ''' Override super().mixer() in order to save the output '''
        self.motor_speed_command = super().mixer(torque_sp, T_sp, logentry)
        return self.motor_speed_command

    def get_features(self, X):
        pass

    def get_f_hat(self, X):
        ''' Returns Force() named tuple with f_hat and, optionally, its derivatives '''
        raise NotImplementedError

    def inner_adapt(self, X, fhat, y):
        raise NotImplementedError

    def update_batch(self, X, fhat, y):
        raise NotImplementedError

    def meta_adapt(self, ):
        raise NotImplementedError


class MetaAdaptBiconvex(MetaAdapt):
    _name = 'biconvex-omac'
    name_long = 'OMAC (bi-convex)'
    def __init__(self, dim_a=100, dim_A=1000, eta_a_base=0.01, eta_A_base=0.01, feature_freq=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_freq = feature_freq
        self.dim_a = dim_a
        self.dim_A = dim_A - dim_A%3
        self.A = None
        self.a = None
        self.W = None
        self.b = None

        # Initialize parameters
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base

        print('Initializing %s with dim_a=%d, dim_A=%d' % (self._name, self.dim_a, self.dim_A))

    def reset_controller(self):
        super().reset_controller()
        # Set W and b matrices
        self.W = np.random.normal(size=(3, self.dim_A, 13)) * self.feature_freq
        self.b = np.random.uniform(low=0, high=np.pi*2, size=(3, self.dim_A))
        # Make W and b block diagonal
        self.W = self.W * np.kron(np.eye(3), np.ones(int(self.dim_A/3)))[:,:,np.newaxis]
        self.b = self.b * np.kron(np.eye(3), np.ones(int(self.dim_A/3)))
        # Make W and b patterned block diagonal
        # _W = np.random.normal(size=(int(self.dim_A/3), 13))
        # self.W = np.kron(np.eye(3), _W)
        # _b = np.random.uniform(low=0, high=np.pi*2, size=int(self.dim_A/3))
        # self.b = np.kron(np.eye(3), _b)

        # Initialize learned parameter matrices
        self.A = np.random.normal(size=(self.dim_A, self.dim_a))
        # self.A /= np.sqrt(self.dim_A) # np.linalg.norm(self.A, 2) / self.dim_A
        self.A /= np.linalg.norm(self.A, 2)
        self.a = np.zeros(self.dim_a)

        # Initialize counters
        self.inner_adapt_count = 0
        self.meta_adapt_count = 0

        self.reset_batch()

    def get_Y(self, X):
        return np.sin(self.W @ X + self.b)

    def get_f_hat(self, X):
        Y = self.get_Y(X)
        return Force(Y @ self.A @ self.a)

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = self.eta_a_base / np.sqrt(self.inner_adapt_count)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_Y(X) @ self.A

    def update_batch(self, X, fhat, y):
        self.batch.append((X, fhat, y, self.a.copy(), self.A.copy()))

    def reset_batch(self):
        self.batch = []

    def meta_adapt(self):
        self.inner_adapt_count = 0
        self.meta_adapt_count += 1
        eta_A = self.eta_A_base / np.sqrt(self.meta_adapt_count)

        A_grad = np.zeros_like(self.A)
        for X, fhat, y, a, A in self.batch:
            Y = self.get_Y(X)
            A_grad += 2 * np.outer((fhat - y).transpose() @ Y, a)
        self.A -= eta_A * A_grad

        self.reset_batch()
        self.meta_adapt_count = 0


class MetaAdaptBaseline(MetaAdaptBiconvex):
    _name = 'baseline-omac'
    name_long = 'Baseline'
    def __init__(self, dim_a=100, dim_A=150, A_type='eye', *args, **kwargs):
        if A_type == 'eye':
            dim_a = dim_a - (dim_a % 3)
            super().__init__(eta_A_base=0.0, dim_A = dim_a, dim_a = dim_a, *args, **kwargs)
        elif A_type == 'random':
            super().__init__(eta_A_base=0.0, dim_A = dim_A, dim_a = dim_a, *args, **kwargs)
        else: 
            raise NotImplementedError
        self.A_type = A_type


    def reset_controller(self):
        super().reset_controller()
        if self.A_type == 'eye':
            self.A = np.eye(self.dim_a)
        # print(self.A.shape)


class MetaAdaptConvex(MetaAdapt):
    _name = 'convex-omac'
    name_long = 'OMAC (convex)'
    def __init__(self, dim_a=100, dim_A=100, eta_a_base=0.001, eta_A_base=0.001, eta_A_threshold=1.0, feature_freq=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_freq = feature_freq
        self.dim_a = dim_a - dim_a%3
        self.dim_A = dim_A - dim_A%3

        # Create variables for kernels and parameters
        self.W_a = None
        self.W_A = None
        self.b_a = None
        self.b_A = None
        self.A = None
        self.a = None

        # Initialize parameters
        self.eta_a_base = eta_a_base
        self.eta_A_threshold = eta_A_threshold
        self.eta_A_base = eta_A_base

        print('Initializing %s with dim_a=%d, dim_A=%d' % (self._name, self.dim_a, self.dim_A))

    def reset_controller(self):
        super().reset_controller()
        # Set W and b matrices
        self.W_a = np.random.normal(size=(3, self.dim_a, 13)) * self.feature_freq
        self.b_a = np.random.uniform(low=0, high=np.pi*2, size=(3, self.dim_a))
        self.W_A = np.random.normal(size=(3, self.dim_A, 13)) * self.feature_freq
        self.b_A = np.random.uniform(low=0, high=np.pi*2, size=(3, self.dim_A))
        # Make W and b block diagonal
        self.W_a = self.W_a * np.kron(np.eye(3), np.ones(int(self.dim_a/3)))[:,:,np.newaxis]
        self.b_a = self.b_a * np.kron(np.eye(3), np.ones(int(self.dim_a/3)))
        self.W_A = self.W_A * np.kron(np.eye(3), np.ones(int(self.dim_A/3)))[:,:,np.newaxis]
        self.b_A = self.b_A * np.kron(np.eye(3), np.ones(int(self.dim_A/3)))

        # Initialize learned parameter matrices
        self.A = np.zeros(self.dim_A)
        self.a = np.zeros(self.dim_a)

        # Set counters
        self.inner_adapt_count = 0
        self.meta_adapt_count = 0

        self.reset_batch()

    def get_Y_a(self, X):
        return np.sin(self.W_a @ X + self.b_a)

    def get_Y_A(self, X):
        return np.sin(self.W_A @ X + self.b_A)

    def get_f_hat(self, X):
        Y_a = self.get_Y_a(X)
        Y_A = self.get_Y_A(X)
        return Force(Y_A @ self.A + Y_a @ self.a)

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = self.eta_a_base / np.sqrt(self.inner_adapt_count)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_Y_a(X)

    def update_batch(self, X, fhat, y):
        self.batch.append((X, fhat, y, self.a.copy(), self.A.copy()))

    def reset_batch(self):
        self.batch = []

    def meta_adapt(self):
        self.inner_adapt_count = 0
        self.meta_adapt_count += 1
        eta_A = min(
            self.eta_A_base / np.sqrt(self.meta_adapt_count),
            self.eta_A_threshold)

        A_grad = np.zeros_like(self.A)
        for X, fhat, y, a, A in self.batch:
            Y_A = self.get_Y_A(X)
            A_grad += 2 * (fhat - y).transpose() @ Y_A
        self.A -= eta_A * A_grad

        self.reset_batch()
        self.meta_adapt_count = 0


class MetaAdaptDeep(MetaAdapt):
    _name = 'deep-omac'
    name_long = 'OMAC (deep)'

    class Phi(nn.Module):

        def __init__(self, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(13, layer_sizes[0]))
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

            return x

    def __init__(self, dim_a=100, eta_a_base=0.001, eta_A_base=0.001, layer_sizes=(25, 30), *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize kernels
        self.dim_a = dim_a - dim_a % 3
        self.phi : self.Phi = None
        self.a : np.ndarray= None
        self.optimizer : optim.Adam = None
        self.loss = nn.MSELoss()
        self.layer_sizes = layer_sizes

        # Initialize parameters
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base

        print('Initializing %s with dim_a=%d, layer_sizes=(%d, %d)' % (self._name, self.dim_a, self.layer_sizes[0], self.layer_sizes[1]))

    def reset_controller(self):
        super().reset_controller()
        self.a = np.zeros(self.dim_a)
        self.phi = self.Phi(dim_kernel=int(self.dim_a / 3), layer_sizes=self.layer_sizes)
        self.optimizer = optim.Adam(self.phi.parameters(), lr=self.eta_A_base)
        self.inner_adapt_count = 0
        self.meta_adapt_count = 0

        self.reset_batch()

    def get_phi(self, X : np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return np.kron(np.eye(3), self.phi(torch.from_numpy(X)).numpy())

    def get_f_hat(self, X):
        phi = self.get_phi(X)
        return Force(phi @ self.a)

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = self.eta_a_base / np.sqrt(self.inner_adapt_count)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_phi(X)

    def update_batch(self, X, fhat, y):
        self.batch.append((X, y, self.a.copy()))

    def reset_batch(self):
        self.batch = []

    def meta_adapt(self):
        self.inner_adapt_count = 0
        # self.meta_adapt_count += 1
        # eta_A = self.eta_A_base / np.sqrt(self.meta_adapt_count)

        self.optimizer.zero_grad()
        loss = 0.
        for X, y, a in self.batch:
            # with torch.no_grad():
            #     z = torch.zeros_like(self.phi(torch.from_numpy(X)))
            # phi 
            phi = torch.kron(torch.eye(3), self.phi(torch.from_numpy(X)))
            loss += self.loss(torch.matmul(phi, torch.from_numpy(a)), torch.from_numpy(y))

        loss.backward()
        self.optimizer.step()
        
        # logentry['loss'] = loss.item()

        self.reset_batch()
        self.meta_adapt_count = 0
        
        
class Omniscient(Baseline):
    _name = 'omniscient'
    name_long = 'Omniscient'
    def __init__(self, *args, **kwargs):
        super().__init__(integral_control=False, *args, **kwargs)
        self.f_hat_m1 = np.zeros(3)
        self.f_hat_dot_filtered = np.zeros(3)

    def reset_controller(self):
        self._first_call = True
        return super().reset_controller()

    def get_Fr(self, X, imu=np.zeros(3), pd=np.zeros(2), vd=np.zeros(2), ad=np.zeros(3), logentry=None, **kwargs):
        # Get baseline controller force (without integral control)
        force = super().get_Fr(X, imu, pd, vd, ad, logentry)
        Fr, Fr_dot, Fr_ddot = force

        # Add in adaptive term
        try:
            f_hat = logentry['Fs']
        except KeyError as err:
            if self._first_call:
                self._first_call = False
            else:
                warn(str(err))
            f_hat = np.zeros(3)

        lam = np.exp(- self.dt / self.params['force_filter_time_const'])
        lam = 0.
        self.f_hat_dot_filtered *= lam
        self.f_hat_dot_filtered += (1 - lam) * (f_hat - self.f_hat_m1) * self.dt_inv

        # f_hat_dot = (f_hat - self.f_hat_m1) * self.dt_inv
        # f_hat_ddot = (f_hat_dot - self.f_hat_dot_m1) / self.params.dt_posctrl
        self.f_hat_m1 = f_hat
        # self.f_hat_dot_m1 = f_hat_dot
        
        # self.f_hat_dot_filtered *= 0.95
        # self.f_hat_dot_filtered += 0.05 * f_hat_dot
        # self.f_hat_ddot *= 0.99
        # self.f_hat_ddot += 0.01 * f_hat_ddot
        # f_hat[2:4] = self.f_hat_dot
        # f_hat[4:6] = self.f_hat_ddot
            
        if logentry is not None:
            logentry['f_hat'] = Force(f_hat)
            # logentry['Fx_r_no_adaptive'] = Fx_r
            # logentry['Fy_r_no_adaptive'] = Fy_r

        Fr -= f_hat
        Fr_dot -= self.f_hat_dot_filtered

        return Force(Fr, Fr_dot, Fr_ddot)