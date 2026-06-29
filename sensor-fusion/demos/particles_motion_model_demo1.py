# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from matplotlib.gridspec import GridSpec
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from numpy.random import uniform, seed
from .lib.robot import robot_draw, Robot
from .lib.pose import Pose
from .lib.utils import gauss


def particles_motion_model_demo1_plot(axes, Xmin=-1, Xmax=1, Ymin=0, Ymax=1,
                                      Thetamin=90, Thetamax=90, Nparticles=10,
                                      muV=1, muOmega=0, sigmaV=0,
                                      sigmaOmega=0, steps=0):

    Thetamin = np.radians(Thetamin)
    Thetamax = np.radians(Thetamax)

    seed(1)

    robots = []
    for m in range(Nparticles):
        robot = Robot(uniform(Xmin, Xmax), uniform(Ymin, Ymax),
                      uniform(Thetamin, Thetamax))
        robots.append(robot)

    ax1, ax2, ax3 = axes
    for ax in axes:
        ax.clear()

    Pose(0, 0, 0).draw_axes(ax1)

    ax1.set_xlim(-5, 5)
    ax1.set_ylim(0, 5)
    ax1.grid(True)

    v = np.linspace(-5, 5, 100)
    ax2.plot(v, gauss(v, muV, sigmaV + 1e-12))
    ax2.set_xlabel('$v$')
    ax2.set_ylabel('$f_V(v)$')
    ax2.set_yticks([])

    omega = np.linspace(-20, 20, 100)
    ax3.plot(omega, gauss(omega, muOmega, sigmaOmega + 1e-12))
    ax3.set_xlabel(r'$\omega$')
    ax3.set_ylabel(r'$f_{\Omega}(\omega)$')
    ax3.set_yticks([])

    for n in range(steps + 1):
        colour = ['red', 'orange', 'green', 'blue', 'magenta'][n % 5]

        for m, robot in enumerate(robots):
            robot_draw(ax1, robot.x, robot.y, robot.heading, colour=colour)
            v = muV + np.random.randn() * sigmaV
            omega = muOmega + np.random.randn() * sigmaOmega

            robot.transition(v, np.radians(omega), dt=1)


def particles_motion_model_demo1():

    fig, axes = subplots(1, figsize=(10, 5))

    gs = GridSpec(5, 4)
    ax1 = fig.add_subplot(gs[0:5, 0:3])
    ax2 = fig.add_subplot(gs[0:2, 3])
    ax3 = fig.add_subplot(gs[3:5, 3])
    axes = [ax1, ax2, ax3]

    show()

    interact(particles_motion_model_demo1_plot, axes=fixed(axes),
             Xmin=(-1, 1, 0.1), Xmax=(-1, 1, 0.1),
             Ymin=(-1, 1, 0.1), Ymax=(-1, 1, 0.1),
             Thetamin=(-180, 180, 15), Thetamax=(-180, 180, 15),
             Nparticles=(10, 100, 10),
             muV=(0, 2, 0.1), muOmega=(-2, 2, 0.1),
             sigmaV=(0.1, 1, 0.1), sigmaOmega=(1, 10, 1),
             steps=(0, 5),
             continuous_update=False)
