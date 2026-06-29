# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.signal_plot import signal_plot

# This shows a non-linear motion model for the x-position of
# a pendulum bob for a pendulum of length l.

def ekf_demo1_plot(axes, l=1.0):

    dt = 0.01

    N = 200
    n = np.arange(N)

    g = 9.81
    theta_0 = np.radians(10)

    # Period of pendulum if theta_0 small
    # T0 = 2 * np.pi * np.sqrt(l / g)
    # print(T0)

    t = n * dt
    theta = theta_0 * np.cos(np.sqrt(g / l) * t)
    # x position of pendulum
    x = l * np.sin(theta)

    signal_plot(t, x, axes=axes)


def ekf_demo1():

    fig, axes = subplots(1)
    show()

    interact(ekf_demo1_plot, axes=fixed(axes), l=(0.1, 2.0, 0.1),
             continuous_update=False)
