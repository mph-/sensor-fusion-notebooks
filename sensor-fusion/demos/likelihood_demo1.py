# M. P. Hayes UCECE
import numpy as np
from ipywidgets import interact, interactive, fixed
from matplotlib.pyplot import subplots
from .lib.utils import gauss

distributions = ['gaussian', 'uniform']


def pdf(x, muX, sigmaX, distribution):

    if distribution == 'gaussian':
        return gauss(x, muX, sigmaX)
    elif distribution == 'uniform':
        xmin = muX - np.sqrt(12) * sigmaX / 2
        xmax = muX + np.sqrt(12) * sigmaX / 2
        return 1.0 * ((x >= xmin) & (x <= xmax)) / (xmax - xmin)
    raise ValueError('Unknown distribution %s' % distribution)

def likelihood_demo1_plot(sigmaV=0.5, z=2,
                          distV=distributions[0]):

    Nx = 801
    x = np.linspace(0.001, 5, Nx)

    h = x

    fZgX = pdf(z - h, 0, sigmaV, distV)

    fig, axes = subplots(2, figsize=(10, 5))
    fig.tight_layout()

    axes[0].plot(x, h, color='orange', label='$h(x) = x$')
    axes[0].plot(np.interp(z, h, x), z, 'o', color='orange')
    axes[0].grid(True)
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$z$')
    axes[0].legend()    

    axes[1].plot(x, fZgX, '--', label='$l_{X|Z}(x|%.1f)$ likelihood' % z)
    axes[1].set_xlabel('$x$')
    axes[1].legend()
    

def likelihood_demo1():
    interact(likelihood_demo1_plot, sigmaV=(0.01, 5, 0.01), z=(0, 5, 0.2),
             distV=distributions)
