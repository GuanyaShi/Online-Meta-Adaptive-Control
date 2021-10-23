from collections import namedtuple
import json

import numpy as np
import rowan

__author__ = "Michael O'Connell"
__date__ = "Octoboer 2021"


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getattr__(self, key):
        return self[key]

def readparamfile(filename, params=None):
    if params is None:
        params = {}
    
    with open(filename) as file:
        params.update(json.load(file))

    return params

_nan_array = np.empty((3,3))
_nan_array[:] = np.nan
_nan_array.flags.writeable = False
State = namedtuple('State', 'p q v w R', defaults=(_nan_array))

def get_state_components(X):
    return State(p=X[0:3], q=X[3:7], v=X[7:10], w=X[10:])

def get_state_components_with_R(X):
    return State(p=X[0:3], q=X[3:7], v=X[7:10], w=X[10:], R=rowan.to_matrix(X[3:7]))


def get_subclass_list(cls):
    names = []
    for c in cls.__subclasses__():
        names.append(c._name)
    return names

def get_subclass(cls, name, **kwargs):
    names = get_subclass_list(cls)
    for c in cls.__subclasses__():
        if name == c._name or name in c._names:
            return c(**kwargs)


Statistics = namedtuple('Statistics', 'count mean rmse max std')

class StatisticsTracker():
    def __init__(self):
        self.reset()

    def update(self, err):
        self.count += 1

        # err = np.abs(err)

        if self.err_mean is None:
            self.err_mean = np.zeros_like(err)
            self.err_max = np.zeros_like(err)
            self.err_squ_cum = np.zeros_like(err)
            self.err_var = np.zeros_like(err)

        self.err_max = np.maximum(err, self.err_max)

        self.err_squ_cum = self.err_squ_cum + err ** 2
        err_mean_old = self.err_mean
        self.err_mean = self.err_mean + (err - self.err_mean) / self.count
        self.err_var = self.err_var + err_mean_old ** 2 - self.err_mean ** 2 + (err ** 2 - self.err_var - err_mean_old ** 2) / self.count

    def get_statistics(self) -> Statistics:
        if self.count > 0:
            return Statistics(self.count, self.err_mean, np.sqrt(self.err_squ_cum / self.count), self.err_max, np.sqrt(self.err_var))
        else:
            return Statistics(0.,0.,0.,0.,0.)

    def reset(self):
        self.count = 0
        self.err_mean = None
        self.err_max = None
        self.err_squ_cum = None
        self.err_var = None

    # def print(self, tag='', freq=1):
    #     if self.count % freq == 0:
    #         count, err_mean, err_rmse, err_max, err_std = self.get_statistics()
    #         print(tag + ' error statistics:' +
    #             '  count=%i' % count +
    #             '  mean=%.2f' % err_mean +
    #             '  rmse=%.2f' % err_mean +
    #             '  max=%.2f' % err_max +
    #             '  std=%.2f' % err_std)


def format_plot(ax):
    ax.margins(x=0)