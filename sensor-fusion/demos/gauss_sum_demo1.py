# M. P. Hayes UCECE
import numpy as np
from ipywidgets import interact, fixed, interactive, fixed
from .lib.signal_plot import signal_plot3
from .lib.utils import gauss

def gauss_sum_demo1_plot(muX=0, sigmaX=1, muY=0, sigmaY=1):

    N = 401
    x = np.linspace(-8, 8, N)

    fX = gauss(x, muX, sigmaX)
    fY = gauss(x, muY, sigmaY)

    muZ = muX + muY
    sigmaZ = np.sqrt(sigmaX**2 + sigmaY**2)
    
    fZ = gauss(x, muZ, sigmaZ)        

    signal_plot3(x, fX, x, fY, x, fZ)

def gauss_sum_demo1():
    interact(gauss_sum_demo1_plot, muX=(-5, 5), muY=(-5, 5),
             sigmaX=(0.01, 5, 0.01), sigmaY=(0.01, 5, 0.01))

    
    

    

