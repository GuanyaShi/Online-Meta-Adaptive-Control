from tokenize import String
import time
from IPython.display import HTML
import os
import shutil
import json

import numpy as np
import matplotlib.pyplot as plt


__author__ = "Michael O'Connell"
__date__ = "Octoboer 2021"
__copyright__ = "Copyright 2021 by Michael O'Connell"
__maintainer__ = "Michael O'Connell"
__email__ = "moc@caltech.edu"
__status__ = "Prototype"


def plot_3d(log, bound=2.5, savename=None, nametag='Quadrotor simulation'):
    fig = plt.figure(figsize=(6,3))
    fig.add_subplot(121, projection='3d')
    plt.plot(log['X'][:,0],log['X'][:,1], log['X'][:,2])
    plt.plot(log['pd'][:,0], log['pd'][:,1], log['pd'][:,2])
    ax = plt.gca()
    ax.set_xlim(-bound,bound)
    ax.set_ylim(-bound,bound)
    ax.set_zlim(-bound,bound)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.add_subplot(122)
    plt.plot(log['X'][:,0],log['X'][:,2])
    plt.plot(log['pd'][:,0], log['pd'][:,2])
    plt.axis((-bound,bound,-bound,bound))
    plt.xlabel('x')
    plt.ylabel('z')
    rmse = np.sqrt(np.mean(np.sum((log['X'][:, 0:3] - log['pd'][:])**2,1)))
    plt.title(nametag + '\nrmse = ' + '%.3fm' % rmse)
    if savename is not None:
        plt.savefig(savename)

def plot_xyz(log, savename=None):
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1)
    plt.plot(log['t'], log['X'][:,0])
    plt.plot(log['t'], log['pd'][:,0])
    plt.legend(('x act', 'x des',))
    plt.xlabel('t')
    plt.ylabel('x')

    plt.subplot(1,3,2)
    plt.title('Actual vs. desired position')
    plt.plot(log['t'], log['X'][:,1])
    plt.plot(log['t'], log['pd'][:,1])
    plt.legend(( 'y act', 'y des'))
    plt.xlabel('t')
    plt.ylabel('y')

    plt.subplot(1,3,3)
    plt.plot(log['t'], log['X'][:,2])
    plt.plot(log['t'], log['pd'][:,2])
    plt.legend(( 'z act', 'z des'))
    plt.xlabel('t')
    plt.ylabel('z')

    if savename is not None:
        plt.savefig(savename)

def plot_error(log, options):
    plt.figure()
    # istart = - int(100 * options['T'])
    istart = 0
    plt.plot(log['t'][istart:], np.sum((log['X'][istart:, 0:3] - log['pd'][istart:,:])**2,1))

    
def plot_fields(data, fields, elements=None, savename=None, axislabels=None):
    ''' data is a dictionary with keys called fields '''
    if elements is None:
        elements = np.arange(data[fields[0]].shape[1])
    num_elements = len(elements)

    # fig = plt.figure(figsize=(5*num_elements, 10))
    fig, axs = plt.subplots(1, num_elements, figsize=(3*num_elements, 3))
    if num_elements == 1:
        axs = [axs,]
    for i in range(num_elements):
        ax = axs[i]
        # ax.plot(num_elements, 1, i+1)
        ax.grid()
        if axislabels is not None:
            suffix = '_' + axislabels[i]
        elif num_elements > 1 and num_elements < 4:
            suffix = '_' + 'xyz'[i]
        else: 
            suffix = ''
        for field in fields:
            ax.plot(data['t'], data[field][:, elements[i]], label=field + suffix)
        ax.legend()
        ax.set_xlabel('t [s]')
    
    if savename is not None:
        plt.savefig(savename)
    # plt.show()
    plt.tight_layout()

    return fig, axs


def get_error(X, pd, istart = 0, iend=-1, lower_percentile=25, upper_percentile=75):
    istart = int(istart)
    iend = int(iend)
    squ_error = np.sum((X[istart:, 0:3] - pd[istart:])**2,1)
    rmse = np.sqrt(np.mean(squ_error))
    meanerr = np.mean(np.sqrt(squ_error))
    maxerr = np.max(np.sqrt(squ_error))
    fifth = np.sqrt(np.percentile(squ_error, lower_percentile))
    ninetyfifth = np.sqrt(np.percentile(squ_error, upper_percentile))
    return dict(rmse=rmse, fifth=fifth, ninetyfifth=ninetyfifth, meanerr=meanerr, maxerr=maxerr)

def save(data, options, testname=None):
    if testname is None:
        testname = options['nametag'] + '_' + options['trajectory']

    folder = 'data/experiments/' + testname + '/'

    if not os.path.isdir('./data/experiments/'):
        os.makedirs('./data/experiments/')
    if os.path.isdir(folder):
        print('Warning: overwriting dataset in folder' + folder)
        # os.rmdir(folder)
        shutil.rmtree(folder)
    os.makedirs(folder)
    print('Created data folder ' + folder)

    with open(folder + 'options.json', 'w') as f:
        json.dump(options, f, indent=4)

    for indexes, log  in np.ndenumerate(data):
        subfolder = folder + log['name'] + '/'
        print('  saving ' + log['name'] + ' to folder ' + subfolder)

        os.makedirs(subfolder)

        for field in log:
            if type(log[field]) is np.ndarray:
                np.save(subfolder + field + '.npy', log[field], allow_pickle=False)
            if type(log[field]) is str:
                with open(subfolder + field + '.txt', 'w') as f:
                    f.write(log[field])

# def load(testname):
#     folder = 'data/experiments/' + testname + '/'

#     with open(folder + 'options.json') as f:
#         options = dict(json.load(f))

#     for expname in os.listdir(folder):
#         if os.path.isdir(expname):
#             for filename in os.listdir(folder + expname):
#                 if filename[-]
#             pass
#         else:
#             pass
#     for i in range(data.options['number_wind_conditions']):
#         subfolder = folder + str(i) + '/'
#         print('  loading wind condition ' + str(i) + ' from folder ' + subfolder)
        
#         data.Meta_X.append(np.load(subfolder + 'X.npy'))
#         data.Meta_Y.append(np.load(subfolder + 'Y.npy'))
#         data.Meta_C.append(np.load(subfolder + 'C.npy'))

#     return data, options