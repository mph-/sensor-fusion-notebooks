# M. P. Hayes UCECE
import numpy as np
from ipywidgets import interact, fixed, interactive, fixed
from .lib.signal_plot import signal_plot

def uniform_sum_demo1_plot(xmin=-1, xmax=1):

    N = 401
    x = np.linspace(-10, 10, N)

    fX1 = 1.0 * ((x >= xmin) & (x <= xmax))
    fX1 /= np.trapz(fX1, x)
    fX2 = 1.0 * ((x >= xmin) & (x <= xmax))
    fX2 /= np.trapz(fX2, x)    

    dx = x[1] - x[0]    
    offset = int(-x[0] / dx)

    fZ = np.convolve(fX1, fX2) * dx

    M = fZ.shape[-1]
    x = np.arange(-M // 2, M // 2) * dx    
    mx = (x < 8) & (x > -8)    
    
    signal_plot(x[mx], fZ[mx])

def uniform_sum_demo1():
    interact(uniform_sum_demo1_plot, xmin=(-5, 5), xmax=(-5, 5), 
             continuous_update=False)
    
    

    

