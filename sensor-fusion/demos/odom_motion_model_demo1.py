# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.robot import robot_draw, Robot2
from .lib.pose import Pose


def odom_motion_model_demo1_plot(axes, d=1, phi1=0, phi2=0, heading=90, steps=1):

    x = np.zeros(steps + 1)
    y = np.zeros(steps + 1)
    theta = np.zeros(steps + 1)

    phi1 = np.radians(phi1)
    phi2 = np.radians(phi2)

    robot = Robot2(heading=np.radians(heading))

    for m in range(steps + 1):
        x[m] = robot.x
        y[m] = robot.y
        theta[m] = robot.heading
        robot.transition(d, phi1, phi2, dt=1)

    fig, ax = subplots(figsize=(10, 5))
    Pose(0, 0, 0).draw_axes(ax)

    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 5)
    ax.grid(True)
    # ax.axis('equal')

    for m in range(len(x)):
        colour = ['red', 'orange', 'green', 'blue', 'magenta'][m % 5]
        robot_draw(ax, x[m], y[m], theta[m], colour=colour)


def odom_motion_model_demo1():


    interact(odom_motion_model_demo1_plot, axes=fixed(axes), d=(0, 2, 0.1),
             phi1=(-180, 180, 15), phi2=(-180, 180, 15),
             steps=(0, 10),
             heading=(0, 180, 15), continuous_update=False)
