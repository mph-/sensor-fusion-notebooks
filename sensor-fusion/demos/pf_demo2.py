# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed, interactive, fixed
from scipy.interpolate import interp1d
from matplotlib.pyplot import figure
from .lib.utils import gauss
from .lib.kde import KDE

distributions = ['uniform', 'gaussian']

def pdf(x, muX, sigmaX, distribution):

    if distribution == 'gaussian':
        return gauss(x, muX, sigmaX)
    elif distribution == 'uniform':
        xmin = muX - np.sqrt(12) * sigmaX / 2
        xmax = muX + np.sqrt(12) * sigmaX / 2
        return 1.0 * ((x >= xmin) & (x <= xmax)) / (xmax - xmin)
    raise ValueError('Unknown distribution %s' % distribution)


def sample(x, fX, M):

    dx = x[1] - x[0]    
    FX = np.cumsum(fX) * dx
    
    interp = interp1d(FX, x, kind='linear', bounds_error=False,
                      fill_value=x[-1])

    return interp(np.random.rand(M))    


def pf_demo2_plot(v=2.0, sigmaX0=0.1, sigmaV=0.4, sigmaW=0.4,
                  seed=1, step=1, M=10, resample=False, kde=True):

    np.random.seed(seed)

    dt = 1    
    A = 1
    B = dt
    C = 1
    D = 0

    Nx = 1000
    x = np.linspace(-10, 40, Nx)
    
    #dist = 'uniform'
    dist = 'gaussian'    
    fX = pdf(x, 0, sigmaX0, dist)

    px_initial = sample(x, fX, M)
    weights_initial = np.ones(M)

    for m in range(1, step + 1):

        if m > 1:
            px_initial = px_posterior
            weights_initial = weights_posterior            
        
        px_prior = px_initial + B * v + np.random.randn(M) * sigmaW
        weights_prior = weights_initial
        
        z = C * m * dt * v + np.random.randn(1) * sigmaV

        px_posterior = px_prior
        weights_posterior = weights_prior * gauss(px_posterior, z, sigmaV)

        if resample:
            fXpostest = KDE(px_posterior, weights_posterior).estimate(x)
            px_posterior = sample(x, fXpostest, M)
            weights_posterior = np.ones(M)            
            

    fig = figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.grid(True)

    width = 0.03
    alpha = 0.5
    ax.bar(px_initial, weights_initial, width=width, linewidth=0, alpha=alpha, label='$X_{%d}$ initial' % (m - 1))

    ax.bar(px_prior, weights_prior, width=width, linewidth=0, alpha=alpha, label='$X_{%d}^{-}$ prior' % m)

    ax.bar(px_posterior, weights_posterior, width=width, linewidth=0, alpha=alpha, label='$X_{%d}^{+}$ posterior' % m)
    ax.set_xlim(-5, 10)
    ax.set_ylim(0, 1.1)

    if kde:
        fXpostest = KDE(px_posterior, weights_posterior).estimate(x)
        fXpriortest = KDE(px_prior, weights_prior).estimate(x)
        fXinitialest = KDE(px_initial, weights_initial).estimate(x)        
        
        ax2 = ax.twinx()
        ax2.plot(x, fXinitialest)
        ax2.plot(x, fXpriortest)
        ax2.plot(x, fXpostest)
    
    ax.legend()

def pf_demo2():
    interact(pf_demo2_plot, step=(1, 5), M=(10, 100, 10),
             v=(1.0, 4.0, 0.2),
             sigmaX0=(0.1, 1, 0.1),
             sigmaV=(0.1, 1, 0.1),
             sigmaW=(0.1, 1, 0.1),
             seed=(1, 100, 1),
             continuous_update=False)
