# Michael P. Hayes UCECE, Copyright 2018--2024
from numpy import radians, exp, linspace, zeros
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.utils import angle_difference
from .lib.robot import Robot


def objective(speed, speed_goal, heading, heading_goal):

    w1 = exp(-abs(speed - speed_goal) / 1.0)
    w2 = exp(-abs(angle_difference(heading, heading_goal)) / radians(30))
    return w1 * w2


def calc_objective(weights, heading, vv, ww, speed_goal, heading_goal, dt):

    ww = radians(ww)
    heading = radians(heading)
    heading_goal = radians(heading_goal)

    for m, w in enumerate(ww):
        for n, v in enumerate(vv):
            robot = Robot(0, 0, heading)
            robot.transition(v, w, dt)
            weights[n, m] = objective(v, speed_goal,
                                      robot.heading, heading_goal)


def dwa_demo1_plot(axes, dt=1, v_max=4, omega_max=360, a_max=2,
                   alpha_max=180, v=1, omega=0, heading=90,
                   speed_goal=1, heading_goal=90):

    w = omega
    w_max = omega_max

    v_min = -v_max
    w_min = -w_max

    v1_max = v + a_max * dt
    v1_min = v - a_max * dt
    w1_max = w + alpha_max * dt
    w1_min = w - alpha_max * dt

    v1_min = max(v_min, v1_min)
    v1_max = min(v_max, v1_max)
    w1_min = max(w_min, w1_min)
    w1_max = min(w_max, w1_max)

    extra_v = v_max * 0.05
    extra_w = w_max * 0.05

    axes.set_xlim(v_min - extra_v, v_max + extra_v)
    axes.set_ylim(w_min - extra_w, w_max + extra_w)

    # Region of all possible speeds
    axes.plot((v_min, v_min, v_max, v_max, v_min),
              (w_min, w_max, w_max, w_min, w_min),
              'b-')

    # Region of all achievable speeds
    axes.plot((v1_min, v1_min, v1_max, v1_max, v1_min),
              (w1_min, w1_max, w1_max, w1_min, w1_min),
              '-', color='orange')

    Nv = 9
    Nw = 9

    weights = zeros((Nv, Nw))

    vv = linspace(v1_min, v1_max, Nv)
    ww = linspace(w1_min, w1_max, Nw)

    calc_objective(weights, heading, vv, ww, speed_goal, heading_goal, dt)

    axes.imshow(weights.T, origin='lower', interpolation=None, aspect='auto',
                extent=(v1_min, v1_max, w1_min, w1_max))

    axes.set_xlabel('$v$')
    axes.set_ylabel(r'$\omega$')

    axes.grid(True)


def dwa_demo1():

    fig, axes = subplots(figsize=(5, 5))
    show()

    interact(dwa_demo1_plot, axes=fixed(axes), dt=(0.1, 2, 0.1),
             v=(0, 2, 0.1), omega=(-60, 60, 15),
             steps=(0, 10), v_max=(1, 5), omega_max=(90, 360, 15),
             a_max=(0.5, 2, 0.5), alpha_max=(30, 180, 15),
             heading=(0, 180, 15), heading_goal=(0, 180, 15),
             speed_goal=(0, 2, 0.1), continuous_update=False)
