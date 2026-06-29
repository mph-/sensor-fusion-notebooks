# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed, interactive, fixed
from matplotlib.pyplot import subplots
from .lib.utils import gauss

def joint_likelihood_demo1_plot(z1=2, z2=4):

    Nx = 801
    x = np.linspace(0.001, 15, Nx)

    var1 = 2**2
    var2 = 0.5**2    

    L1 = gauss(x - z1, 0, np.sqrt(var1))
    L2 = gauss(x - z2, 0, np.sqrt(var2)) + gauss(x - z2 - 8, 0, np.sqrt(var2))

    L = L1 * L2
    
    fig, axes = subplots(1, figsize=(10, 3))
    fig.tight_layout()

    axes.plot(x, L1, '-.', label='$L_1(x|%.0f)$' % z1)

    axes.plot(x, L2, '--', label='$L_2(x|%.0f)$' % z2)

    axes.plot(x, L, '-', label='$L(x|%.0f,%.0f)$' % (z1, z2))

    axes.grid(True)    
    axes.set_xlabel('$x$')
    axes.legend()        


def joint_likelihood_demo1():
    interact(joint_likelihood_demo1_plot, z1=(0, 10, 1), z2=(0, 10, 1))

