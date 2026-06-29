# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact
from matplotlib.pyplot import subplots, show
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


def likelihood_demo5_plot(sigmaV=0.5, z=2,
                          distV=distributions[0], show_Lambda=True):

    Nx = 801
    x = np.linspace(0.1, 20, Nx)

    h = 0.1 * x + 20 / x

    fZgX = pdf(z - h, 0, sigmaV, distV)

    # FIXME Dynamically change number of axes.  This creates
    # a new figure for every call
    fig, axes = subplots(2 + show_Lambda * 1, figsize=(10, 5))
    fig.tight_layout()
    show()

    axes[0].plot(x, h, color='orange', label='$h(x) = 1/x$')
    axes[0].grid(True)
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$z$')
    axes[0].legend()

    axes[1].plot(x, fZgX, '--', label='$L(x|%.1f)$' % z)
    axes[1].set_xlabel('$x$')
    axes[1].grid(True)
    axes[1].legend()

    if show_Lambda:
        zv = np.linspace(0, 5, 201)

        X, Z = np.meshgrid(x, zv)
        H = 1 / X
        Lambda = pdf(Z - H, 0, sigmaV, distV)

        axes[2].imshow(Lambda, origin='lower',
                       extent=(x[0], x[-1], zv[0], zv[-1]))
        axes[2].axis('tight')
        axes[2].set_xlabel('$x$')
        axes[2].set_ylabel('$z$')


def likelihood_demo5():
    interact(likelihood_demo5_plot, sigmaV=(0.01, 5, 0.01), z=(0, 5, 0.2),
             distV=distributions)
