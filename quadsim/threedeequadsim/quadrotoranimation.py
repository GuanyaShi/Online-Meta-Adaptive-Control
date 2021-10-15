import collections
import pkg_resources

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import rowan

from .utils import readparamfile


DEFAULT_QUAD_PARAMETER_FILE = pkg_resources.resource_filename(__name__, 'params/quadrotor.json')

class QuadrotorAnimation():
    LW = 3
    Circle = collections.namedtuple('Circle', 'center radius normal ax1 ax2', defaults=((0., 0., 0.), (0., 0., 0.)))
    def __init__(self):
        self.line_objs = []
        self.circle_objs = []
        self.annotation_objs = []
        params = readparamfile(DEFAULT_QUAD_PARAMETER_FILE)
        l_arm = params['l_arm']
        D = 0.45 # params['D'] * 3
        h = 0.10 # params['h']
        self.lines = (((0.,0.,0.),(l_arm,l_arm,0.)),
                      ((0.,0.,0.),(-l_arm,l_arm,0.)),
                      ((0.,0.,0.),(-l_arm,-l_arm,0.)),
                      ((0.,0.,0.),(l_arm,-l_arm,0.)),
                      ((l_arm,l_arm,0.),(l_arm,l_arm,h)),
                      ((-l_arm,l_arm,0.),(-l_arm,l_arm,h)),
                      ((-l_arm,-l_arm,0.),(-l_arm,-l_arm,h)),
                      ((l_arm,-l_arm,0.),(l_arm,-l_arm,h)),
                     )
        # Make circles mutable so that we can update the ax1 and ax2 fields
        self.circles = [self.Circle((l_arm, l_arm, h), D/2, (0., 0., 1.)),
                        self.Circle((-l_arm, l_arm, h), D/2, (0., 0., 1.)),
                        self.Circle((-l_arm, -l_arm, h), D/2, (0., 0., 1.)),
                        self.Circle((l_arm, -l_arm, h), D/2, (0., 0., 1.)),]
        self.fps = 30
        self.Rs = None 
        self.ps = None 

    def _init_lines(self, ax):
        self.line_objs = []
        for line in self.lines:
            line_obj, = ax.plot([], [], [], 'k-', lw=self.LW)
            self.line_objs.append(line_obj) # Mutable!

    def _update_lines(self, R, p):
        for line_obj, line in zip(self.line_objs, self.lines):
            x, y, z = R @ np.array(line).transpose() + p[:, np.newaxis]
            line_obj.set_data(x, y)
            line_obj.set_3d_properties(z)
        return self.line_objs

    def _init_circles(self, ax):
        self.circle_objs = []
        for i, (center, radius, normal, ax1, ax2) in enumerate(self.circles):
            circle_obj, = ax.plot([], [], [], 'b-', lw=self.LW)
            self.circle_objs.append(circle_obj) # mutable!

            ax1 = np.cross(normal, (1., 0., 0.))
            if np.linalg.norm(ax1) < 1e-3:
                ax1 = np.cross(normal, (0., 0., 1.))
            ax1 = ax1 / np.linalg.norm(ax1)
            ax2 = np.cross(normal, ax1)
            print(np.linalg.norm(normal), np.linalg.norm(ax1), np.linalg.norm(ax2))

            self.circles[i] = self.Circle(np.array(center), radius, np.array(normal), np.array(ax1), np.array(ax2))

    def _update_circles(self, R, p):
        for (center, radius, normal, ax1, ax2), circle_obj in zip(self.circles, self.circle_objs):
            # normal = R @ np.array(normal)
            center = R @ center
            ax1 = R @ ax1
            ax2 = R @ ax2
            theta = np.linspace(0, 2*np.pi)
            x, y, z = radius * np.outer(ax1, np.cos(theta)) + radius * np.outer(ax2, np.sin(theta)) + center[:, np.newaxis] + p[:, np.newaxis]
            circle_obj.set_data(x, y)
            circle_obj.set_3d_properties(z)
        return self.circle_objs

    def _init_annotations(self, ax):
        self.annotation_objs.append(ax.text(1, 1, 1, 't=', transform=ax.transAxes))

    def _update_annotations(self, n):
        self.annotation_objs[0].set_text('t=%.2f' % ((n+1) / self.fps))
        return self.annotation_objs

    def _init_func(self, ax):
        self._init_lines(ax)
        self._init_circles(ax)
        self._init_annotations(ax)

    def _animate_func(self, n):
        i = int(n / self.fps / self.dt)
        R = np.array(self.Rs[i])
        p = np.array(self.ps[i])

        objs = []
        objs.append(self._update_lines(R, p))
        objs.append(self._update_circles(R, p))
        objs.append(self._update_annotations(n))

        return objs
    
    def animate(self, qs, ps, dt, fig=None, ax=None):
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        elif ax is None:
            ax = fig.add_subplot(111, projection='3d')

            
        self.Rs = []
        for q in qs:
            R = rowan.to_matrix(q)
            self.Rs.append(R)

        self.ps = ps 
        self.dt = dt

        xs, ys, zs = np.array(ps).transpose()
        xmid = (min(xs) + max(xs))/2
        ymid = (min(ys) + max(ys))/2
        zmid = (min(zs) + max(zs))/2
        xwid = max(xs) - min(xs)
        ywid = max(ys) - min(ys)
        zwid = max(zs) - min(zs)
        margin = 0.5
        wid = max((xwid, ywid, zwid)) + 2*margin
        ax.set_xlim(xmid - wid/2, xmid + wid/2)
        ax.set_ylim(ymid - wid/2, ymid + wid/2)
        ax.set_zlim(zmid - wid/2, zmid + wid/2)
        
        n = int(len(self.Rs) * self.fps * self.dt)
        
        self._init_func(ax)
        return animation.FuncAnimation(fig, self._animate_func, frames=n, interval = 1000 / self.fps)

    def draw_arrow(self, ax, position, velocity):
        ax.quiver(position[0], position[1], position[2], velocity[0], velocity[1], velocity[2])

    def draw_frames(self, qs, ps, dt, fig=None, ax=None):
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        elif ax is None:
            ax = fig.add_subplot(111, projection='3d')

            
        self.Rs = []
        for q in qs:
            R = rowan.to_matrix(q)
            self.Rs.append(R)

        self.ps = ps 
        self.dt = dt

        xs, ys, zs = np.array(ps).transpose()
        xmid = (min(xs) + max(xs))/2
        ymid = (min(ys) + max(ys))/2
        zmid = (min(zs) + max(zs))/2
        xwid = max(xs) - min(xs)
        ywid = max(ys) - min(ys)
        zwid = max(zs) - min(zs)
        margin = 0.5
        wid = max((xwid, ywid, zwid)) + 2*margin
        ax.set_xlim(xmid - wid/2, xmid + wid/2)
        ax.set_ylim(ymid - wid/2, ymid + wid/2)
        ax.set_zlim(zmid - wid/2, zmid + wid/2)
        
        n = int(len(self.Rs) * self.fps * self.dt)
        
        self._init_func(ax)

        for frame in range(n):
            self._animate_func(frame)
            self.draw_arrow(ax, (0,0,0), (np.sin(n),np.cos(n),1))
            plt.savefig(('animate-folder/%04d' % frame) + '.png')


def draw_frames(data, **kwargs):
    a = QuadrotorAnimation()
    ps = data['X'][:, 0:3]
    qs = data['X'][:, 3:7]
    dt = np.mean(data['t'][1:] - data['t'][:-1])
    # dt = data.metadata.quad_params.dt_readout
    return a.draw_frames(qs, ps, dt, **kwargs)

def animate(data, **kwargs):
    a = QuadrotorAnimation()
    ps = data.X[:, 0:3]
    qs = data.X[:, 3:7]
    dt = data.metadata.quad_params.dt_readout
    return a.animate(qs, ps, dt, **kwargs)