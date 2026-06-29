# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from matplotlib.gridspec import GridSpec
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from matplotlib.patches import Arc
from .lib.utils import gauss, wraptopi


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
        ydx = size * np.cos(theta + np.pi/2)
        ydy = size * np.sin(theta + np.pi/2)

        axes.plot(x, y, marker, color=colour, label=label, markersize=10)

        axes.plot((x, x + xdx), (y, y + xdy), color='red', linewidth=3)
        axes.plot((x, x + ydx), (y, y + ydy), color='green', linewidth=3)
        if name is not None:
            axes.text(x + 0.5, y - 0.5, name)


def mvpf_demo2_plot(axes, beacon_x=15, beacon_y=8, beacon_theta=-75,
                    robot_x=3, robot_y=1, robot_theta=15,
                    particle_x=10, particle_y=3, particle_theta=15):

    sigmaDR = 0.5
    sigmaDP = 0.5

    robot = Beacon(robot_x, robot_y, np.radians(robot_theta), 1)
    particle = Beacon(particle_x, particle_y, np.radians(particle_theta), 1)
    beacon = Beacon(beacon_x, beacon_y, np.radians(beacon_theta), 1)

    ax1, ax2, ax3 = axes
    for ax in axes:
        ax.clear()

    ax1.grid(True)
    ax1.axis('scaled')
    ax1.set_xlim(-0.05, 20)
    ax1.set_ylim(-0.05, 10)
    ax1.set_xticks((0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20))

    robot.plot(ax1, marker='p', colour='black', size=5, name='robot')
    particle.plot(ax1, marker='p', colour='black', size=5, name='particle')
    beacon.plot(ax1, name='beacon')

    r = np.sqrt((robot.x - beacon.x)**2 + (robot.y - beacon.y)**2)
    psi = np.arctan2((beacon.y - robot.y), (beacon.x - robot.x))
    phi = wraptopi(psi - robot.theta)

    rp = np.sqrt((particle.x - beacon.x)**2 + (particle.y - beacon.y)**2)
    psip = np.arctan2((beacon.y - particle.y), (beacon.x - particle.x))
    phip = wraptopi(psip - particle.theta)

    ax1.plot([robot.x, beacon.x], [robot.y, beacon.y], '--k')
    ax1.plot([particle.x, beacon.x], [particle.y, beacon.y], '-.k')

    arc = Arc((robot.x, robot.y), 5, 5,
              theta1=np.degrees(robot.theta),
              theta2=np.degrees(psi))
    ax1.add_patch(arc)

    arc = Arc((particle.x, particle.y), 5, 5,
              theta1=np.degrees(particle.theta),
              theta2=np.degrees(psip))
    ax1.add_patch(arc)

    ax1.plot((0, 20), (0, 0), color='red', linewidth=3)
    ax1.plot((0, 0), (0, 10), color='green', linewidth=3)

    dr = r - rp
    dphi = wraptopi(phi - phip)
    a = gauss(dphi / sigmaDP) * gauss(dr / sigmaDR)

    vdr = np.linspace(-10, 10, 100)
    ax2.plot(vdr, gauss(vdr, 0, sigmaDR))
    ax2.plot(dr,  gauss(dr, 0, sigmaDR), 'o')
    ax2.set_xlabel(r'$\Delta r$')
    ax2.set_ylabel(r'$f_{\Delta R}(\Delta r)$')
    ax2.set_yticks([])

    vdp = np.linspace(-np.pi, np.pi, 100)
    ax3.plot(np.degrees(vdp), gauss(vdp, 0, sigmaDP))
    ax3.plot(np.degrees(dphi),  gauss(dphi, 0, sigmaDP), 'o')
    ax3.set_xlabel(r'$\Delta \phi$')
    ax3.set_ylabel(r'$f_{\Delta \Phi}(\Delta \phi)$')
    ax3.set_yticks([])

    ax1.set_title(r'$\Delta r=%.1f, \Delta \phi=%.1f, a=%.3e$' %
                  (dr, np.degrees(dphi), a))


def mvpf_demo2():

    fig, axes = subplots(1, figsize=(12, 5))

    gs = GridSpec(5, 4)
    ax1 = fig.add_subplot(gs[0:5, 0:3])
    ax2 = fig.add_subplot(gs[0:2, 3])
    ax3 = fig.add_subplot(gs[3:5, 3])

    axes = [ax1, ax2, ax3]

    show()

    interact(mvpf_demo2_plot, axes=fixed(axes), robot_x=(0, 20, 1),
             robot_y=(0, 10, 1), robot_theta=(-180, 180, 15),
             beacon_x=(0, 20, 1), beacon_y=(0, 10, 1),
             beacon_theta=(-180, 180, 15), particle_x=(0, 20, 1),
             particle_y=(0, 10, 1), particle_theta=(-180, 180, 15),
             continuous_update=False)
