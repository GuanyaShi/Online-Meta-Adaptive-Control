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