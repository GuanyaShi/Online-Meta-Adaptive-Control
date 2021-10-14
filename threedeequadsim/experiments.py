from tokenize import String
import time
from IPython.display import HTML
import os
import shutil
import json

import numpy as np
import matplotlib.pyplot as plt

from threedeequadsim import quadsim, controller, trajectory, quadrotoranimation, utils

def run(Ctrls, options, master_seed=115):
    # Logs_ijk
    # - i: controller
    # - j: parameter
    Logs = []
    for i, c in enumerate(Ctrls):
        np.random.seed(master_seed)
        Logs.append([])
        for j, value in enumerate(options['parameter_values']):
            # Set up quadrotor object and trajectory object
            t = trajectory.get_trajectory(options['trajectory'], **options['trajectory_options'])
            parameter_passer = {options['parameter_name: value']}
            q = quadsim.QuadrotorWithSideForce(**parameter_passer, **options['q_options'])
            q.params.t_stop = options['simulation_time']

            # set some parameters for the simulation
            experiment_seed = np.random.randint(low=0, high=2147483648)

            # Run experiment
            print('Testing %s with ' % c._name, parameter_passer)
            time.sleep(0.5)
            data = q.run(trajectory=t, controller=c, seed=experiment_seed)
            log, metadata = data

            #
            value_str = str(value).replace('.', 'd').replace(', ', '-')
            for char in "[]{}()'":
                value_str = value_str.replace(char, '')
            log.name = c._name + '_' + options['parameter_name'] + '-' + value_str
            log.seed = experiment_seed

            Logs[i].append(log)
    return Logs

def plot_3d(log, options):
    fig = plt.figure(figsize=(10,5))
    fig.add_subplot(121, projection='3d')
    plt.plot(log['X'][:,0],log['X'][:,1], log.X[:,2])
    plt.plot(log['pd'][:,0], log['pd'][:,1], log.pd[:,2])
    ax = plt.gca()
    bound = 2.5
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
    # tstart = 15.
    # istart = int(100*tstart)
    istart = - int(100 * options['T'])
    rmse = np.sqrt(np.mean(np.sum((log['X'][istart:, 0:3] - log['pd'][istart:])**2,1)))
    print('rmse = ', rmse)
    plt.title('rmspe = ' + '%.3fm' % rmse + ', ' + options['nametag_short'])
    # plt.savefig('tracking-error_' + nametag + '.png')
    plt.savefig('plots/' + options['nametag'] + '_plot-3D.jpg')

def plot_xyz(log, options):
    plt.figure(figsize=(15,5))
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

    plt.savefig('plots/' + options['nametag'] + '_plot-xyz.jpg')

def plot_error(log, options):
    plt.figure()
    # istart = - int(100 * options['T'])
    istart = 0
    plt.plot(log['t'][istart:], np.sum((log['X'][istart:, 0:3] - log['pd'][istart:,:])**2,1))

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
        subfolder = folder + log.name + '/'
        print('  saving ' + log.name + ' to folder ' + subfolder)

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