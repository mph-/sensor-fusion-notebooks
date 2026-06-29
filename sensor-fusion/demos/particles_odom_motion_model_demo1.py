# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from matplotlib.gridspec import GridSpec
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from numpy.random import uniform, seed
from .lib.robot import robot_draw, Robot2
from .lib.pose import Pose
from .lib.utils import gauss


def particles_odom_motion_model_demo1_plot(axes, Xmin=-1, Xmax=1, Ymin=0,
                                           Ymax=1, Thetamin=90,
                                           ThetaMax=90, Nparticles=10,
                                           dp=1, phi1p=0, phi2p=0,
                                           sigmaD=0, sigmaPhi1=0,
                                           sigmaPhi2=0, steps=0):
    muD = 0
    muPhi1 = 0
    muPhi2 = 0
    Thetamin = np.radians(Thetamin)
    ThetaMax = np.radians(ThetaMax)

    seed(1)

    robots = []
    for m in range(Nparticles):
        robot = Robot2(uniform(Xmin, Xmax), uniform(Ymin, Ymax),
                       uniform(Thetamin, ThetaMax))
        robots.append(robot)

    ax1, ax2, ax3, ax4 = axes
    for ax in axes:
        ax.clear()

    Pose(0, 0, 0).draw_axes(ax1)

    ax1.set_xlim(-5, 5)
    ax1.set_ylim(0, 5)
    ax1.grid(True)

    for n in range(steps + 1):
        colour = ['red', 'orange', 'green', 'blue', 'magenta'][n % 5]

        for m, robot in enumerate(robots):
            robot_draw(ax1, robot.x, robot.y, robot.heading, colour=colour)
            d = dp + np.random.randn() * sigmaD
            phi1 = phi1p + np.random.randn() * sigmaPhi1
            phi2 = phi2p + np.random.randn() * sigmaPhi2

            robot.transition(d, np.radians(phi1), np.radians(phi2), dt=1)

    d = np.linspace(-5, 5, 100)
    ax2.plot(d, gauss(d, muD, sigmaD + 1e-12))
    ax2.set_xlabel("$d-d'$")
    ax2.set_ylabel('$f_D(d)$')
    ax2.set_yticks([])

    phi1 = np.linspace(-20, 20, 100)
    ax3.plot(phi1, gauss(phi1, muPhi1, sigmaPhi1 + 1e-12))
    ax3.set_xlabel(r"$\phi_1-\phi_1'$")
    ax3.set_ylabel(r'$f_{\Phi_1}(\phi_1)$')
    ax3.set_yticks([])

    phi2 = np.linspace(-20, 20, 100)
    ax4.plot(phi2, gauss(phi2, muPhi2, sigmaPhi2 + 1e-12))
    ax4.set_xlabel(r"$\phi_2-\phi_2'$")
    ax4.set_ylabel(r'$f_{\Phi_2}(\phi_2)$')
    ax4.set_yticks([])


def particles_odom_motion_model_demo1():

    fig, axes = subplots(figsize=(12, 5))
    gs = GridSpec(8, 4)
    ax1 = fig.add_subplot(gs[0:8, 0:3])
    ax2 = fig.add_subplot(gs[0:2, 3])
    ax3 = fig.add_subplot(gs[3:5, 3])
    ax4 = fig.add_subplot(gs[6:8, 3])

    axes = [ax1, ax2, ax3, ax4]

    show()

    interact(particles_odom_motion_model_demo1_plot, axes=fixed(axes),
             Xmin=(-1, 1, 0.1), Xmax=(-1, 1, 0.1),
             Ymin=(-1, 1, 0.1), Ymax=(-1, 1, 0.1),
             Phimin=(-180, 180, 15), PhiMax=(-180, 180, 15),
             Nparticles=(10, 100, 10),
             dp=(0, 2, 0.1), sigmaD=(0.1, 1, 0.1),
             phi1p=(-2, 2, 0.1), sigmaPhi1=(1, 10, 1),
             phi2p=(-2, 2, 0.1), sigmaPhi2=(1, 10, 1),
             steps=(0, 5),
             continuous_update=False)
