# M. P. Hayes UCECE
import numpy as np
from ipywidgets import interact, fixed, interactive, fixed
from .lib.signal_plot import signal_plot
import mcerp
mcerp.npts = 1000000
from mcerp import N, U

def monte_carlo_demo1_plot():

    D1 = U(0.499, 0.501)
    D2 = U(0.998, 1.002)    
    T = N(0.3e-3, 1e-6)
    V = (D2 - D1) / T

    V.plot()

def monte_carlo_demo1():
    interact(monte_carlo_demo1_plot, continuous_update=False)
    
    

    

