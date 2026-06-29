# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed, interactive, fixed
from matplotlib.pyplot import figure
from .lib.signal_plot import signal_plot2
from .lib.utils import gauss

def kf_demo1_plot(steps=0, v=1, A=1, B=1, C=1, sigmaX0=1, sigmaV=1, sigmaW=1):

    dt = 1
    
    Nx = 801
    x = np.linspace(-10, 40, Nx)
    v = np.linspace(-10, 40, Nx)
    
    dx = x[1] - x[0]    
    offset = int(-x[0] / dx)

    muW = v * dt
    
    fV = gauss(v, 0, sigmaV)

    fW = gauss(x, muW, sigmaW)

    fX = gauss(x, 0, sigmaX0)    

    fig = figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.grid(True)

    mx = (x < 12) & (x > -2)

    xest = 0
    sigmaX = sigmaX0
    
    for m in range(steps + 1):

        if m > 0:
            fX = np.convolve(fX, fW)[offset:offset + len(x)] * dx

        ax.plot(x[mx], fX[mx], label='%d' % m)

    ax.legend()

def kf_demo1():
    interact(kf_demo1_plot, steps=(0, 5), 
             X0=beliefs, Wn=beliefs, continuous_update=False)
