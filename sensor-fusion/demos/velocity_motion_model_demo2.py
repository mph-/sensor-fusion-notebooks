# Michael P. Hayes UCECE, Copyright 2022
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.robot import robot_draw, Robot
from .lib.pose import Pose


def velocity_motion_model_demo2_plot(axes, x0=3, y0=1, heading0=90,
                                     v=1, omega=0, steps=0):

    x = np.zeros(steps + 1)
    y = np.zeros(steps + 1)
    theta = np.zeros(steps + 1)

    xl = x * 0
    yl = y * 0
    thetal = theta * 0

    robot = Robot(x0, y0, heading=np.radians(heading0))
    robotl = Robot(0, 0, 0)

    for m in range(steps + 1):
        x[m] = robot.x
        y[m] = robot.y
        theta[m] = robot.heading
        robot.transition(v, np.radians(omega), dt=1)
        xl[m] = robotl.x
        yl[m] = robotl.y
        thetal[m] = robotl.heading
        robotl.transition(v, np.radians(omega), dt=1)

    for ax in axes:
        ax.clear()

    Pose(0, 0, 0).draw_axes(axes[0])
    Pose(x0, y0, np.radians(heading0)).draw_axes(axes[0], linestyle=':')

    axes[0].set_xlim(-1, 5)
    axes[0].set_ylim(-1, 5)
    # axes[0].axis('equal')
    axes[0].grid(True)
    axes[0].set_title('Global')

    for m in range(len(x)):
        colour = ['red', 'orange', 'green', 'blue', 'magenta'][m % 5]
        robot_draw(axes[0], x[m], y[m], theta[m], colour=colour)

    Pose(0, 0, 0).draw_axes(axes[1])

    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-1, 5)
    # axes[1].axis('equal')
    axes[1].grid(True)
    axes[1].set_title('Local')

    for m in range(len(x)):
        colour = ['red', 'orange', 'green', 'blue', 'magenta'][m % 5]
        robot_draw(axes[1], xl[m], yl[m], thetal[m], colour=colour)


def velocity_motion_model_demo2():

    fig, axes = subplots(1, 2, figsize=(11, 5))
    show()

    interact(velocity_motion_model_demo2_plot, axes=fixed(axes),
             x0=(-4, 4, 0.5),
             y0=(-4, 4, 0.5),
             heading0=(0, 180, 15),
             v=(0, 2, 0.1),
             omega=(-60, 60, 15),
             steps=(0, 10),
             heading=(0, 180, 15),
             continuous_update=False)
