# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from matplotlib.cm import viridis
from matplotlib import colors
from matplotlib.patches import Arc
from .lib.utils import gauss, wraptopi, angle_difference


class Beacon(object):

    def __init__(self, x, y, theta, num=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.num = num

    def plot(self, axes, marker='o', colour='blue', label=None, size=1,
             name=None):

        x, y, theta = self.x, self.y, self.theta

        xdx = size * np.cos(theta)
        xdy = size * np.sin(theta)
        ydx = size * np.cos(theta + np.pi / 2)
        ydy = size * np.sin(theta + np.pi / 2)

        axes.plot(x, y, marker, color=colour, label=label, markersize=10)

        axes.plot((x, x + xdx), (y, y + xdy), color='red', linewidth=3)
        axes.plot((x, x + ydx), (y, y + ydy), color='green', linewidth=3)
        if name is not None:
            axes.text(x + 0.5, y - 0.5, name)


def mvpf_demo3_plot(axes, robot_x=3, robot_y=1, robot_theta=15,
                    particle_theta=15, sigma_dR=0.5, sigma_dPhi=0.1,
                    show_range_weight=True, show_bearing_weight=False):

    beacon_x = 16
    beacon_y = 8
    beacon_theta = -75

    robot = Beacon(robot_x, robot_y, np.radians(robot_theta), 1)
    beacon = Beacon(beacon_x, beacon_y, np.radians(beacon_theta), 1)

    axes.clear()
    axes.grid(True)
    axes.axis('scaled')
    axes.set_xlim(-0.05, 20)
    axes.set_ylim(-0.05, 10)
    axes.set_xticks((0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20))

    robot.plot(axes, marker='p', colour='black', size=5, name='robot')
    beacon.plot(axes, name='beacon')

    r = np.sqrt((robot.x - beacon.x)**2 + (robot.y - beacon.y)**2)
    psi = np.arctan2((beacon.y - robot.y), (beacon.x - robot.x))
    phi = wraptopi(psi - robot.theta)

    axes.plot([robot.x, beacon.x], [robot.y, beacon.y], '--k')

    arc = Arc((robot.x, robot.y), 5, 5,
              theta1=np.degrees(robot.theta),
              theta2=np.degrees(psi))
    axes.add_patch(arc)

    axes.plot((0, 20), (0, 0), color='red', linewidth=3)
    axes.plot((0, 0), (0, 10), color='green', linewidth=3)

    vx = np.linspace(-0.05, 20, 201)
    vy = np.linspace(-0.05, 10, 101)
    Vx, Vy = np.meshgrid(vx, vy)

    Vr = np.sqrt((Vx - beacon.x)**2 + (Vy - beacon.y)**2)
    Vpsi = np.arctan2(beacon.y - Vy, beacon.x - Vx)
    Vphi = wraptopi(Vpsi - np.radians(particle_theta))

    Vdphi = angle_difference(Vphi, phi)

    ar = gauss(Vr - r, 0, sigma_dR)
    ap = gauss(Vdphi, 0, sigma_dPhi)

    upper = viridis(np.arange(256))
    lower = np.ones((int(256 / 4), 4))
    for i in range(3):
        lower[:, i] = np.linspace(1, upper[0, i], lower.shape[0])

    cmap = np.vstack((lower, upper))
    cmap = colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

    if show_range_weight and not show_bearing_weight:
        axes.imshow(ar, origin='lower',
                    extent=(vx[0], vx[-1], vy[0], vy[-1]), cmap=cmap)

    if not show_range_weight and show_bearing_weight:
        axes.imshow(ap, origin='lower',
                    extent=(vx[0], vx[-1], vy[0], vy[-1]), cmap=cmap)

    if show_range_weight and show_bearing_weight:
        axes.imshow(ar * ap, origin='lower',
                    extent=(vx[0], vx[-1], vy[0], vy[-1]), cmap=cmap)

    axes.set_title(r'$r=%.1f, \phi=%.1f$' % (r, np.degrees(phi)))


def mvpf_demo3():

    fig, axes = subplots(1, figsize=(12, 5))
    show()

    interact(mvpf_demo3_plot, axes=fixed(axes), beacon_x=(0, 20, 1),
             beacon_y=(0, 10, 1), beacon_theta=(-180, 180, 15),
             sigma_dR=(0, 1, 0.1), sigma_dPhi=(0.05, 0.5, 0.05),
             particle_theta=(-180, 180, 15), continuous_update=False)
