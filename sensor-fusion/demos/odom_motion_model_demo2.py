# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.robot import robot_draw, Robot2
from .lib.pose import Pose


def odom_motion_model_demo2_plot(axes, x0=3, y0=1, heading0=135,
                                 d=1, phi1=0, phi2=0, steps=1):

    x = np.zeros(steps + 1)
    y = np.zeros(steps + 1)
    theta = np.zeros(steps + 1)

    phi1 = np.radians(phi1)
    phi2 = np.radians(phi2)

    robot = Robot2(x0, y0, heading=np.radians(heading0))

    for m in range(steps + 1):
        x[m] = robot.x
        y[m] = robot.y
        theta[m] = robot.heading
        robot.transition(d, phi1, phi2, dt=1)

    axes.clear()
    Pose(0, 0, 0).draw_axes(axes)
    Pose(x0, y0, np.radians(heading0)).draw_axes(axes, linestyle=':')

    axes.set_xlim(-5, 5)
    axes.set_ylim(0, 5)
    axes.grid(True)
    # axes.axis('equal')

    for m in range(len(x)):
        colour = ['red', 'orange', 'green', 'blue', 'magenta'][m % 5]
        robot_draw(axes, x[m], y[m], theta[m], colour=colour)


def odom_motion_model_demo2():

    fig, axes = subplots(figsize=(10, 5))
    show()

    interact(odom_motion_model_demo2_plot, axes=fixed(axes),
             x0=(-4, 4, 0.5),
             y0=(-4, 4, 0.5),
             heading0=(0, 180, 15),
             d=(0, 2, 0.1),
             phi1=(-180, 180, 15),
             phi2=(-180, 180, 15),
             steps=(0, 10),
             continuous_update=False)
