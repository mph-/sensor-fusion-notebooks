# Michael P. Hayes UCECE, Copyright 2018--2023
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from .lib.utils import gauss

distributions = ['uniform', 'gaussian']


def pdf(x, muX, sigmaX, distribution):

    if distribution == 'gaussian':
        return gauss(x, muX, sigmaX)
    elif distribution == 'uniform':
        xmin = muX - np.sqrt(12) * sigmaX / 2
        xmax = muX + np.sqrt(12) * sigmaX / 2
        return 1.0 * ((x >= xmin) & (x <= xmax)) / (xmax - xmin)
    raise ValueError('Unknown distribution %s' % distribution)


def hf_demo2_plot(axes, distX0='gaussian', sigmaX0=0.4, sigmaV=0.1,
                  sigmaW=0.1, seed=1, M=50):

    np.random.seed(seed)

    v = 2
    dt = 1
    step = 1

    # Tweak number of bins so that user has expected number of
    # bins spanning the plotted range for x.
    M = int(M * (4 - -2) / (3 - -1))

    x = np.linspace(-2, 4, M)
    dx = x[1] - x[0]

    fX_initial = pdf(x, 0, sigmaX0, distX0)
    fX_prior = np.zeros(M)
    fX_posterior = np.zeros(M)

    for m in range(1, step + 1):

        if m > 1:
            fX_initial = fX_posterior

        # Predict step: "Convolve" initial belief histogram with conditional
        # PDF for process model
        for i in range(len(x)):
            total = 0
            for j in range(len(x)):
                total += fX_initial[j] * \
                    gauss(x[i] - x[j] - v * dt, 0, sigmaW)
            fX_prior[i] = total * dx

        z = m * dt * v + np.random.randn(1) * sigmaV

        # Update step: Apply Bayes' theorem
        fX_posterior = fX_prior * gauss(z - x, 0, sigmaV)
        fX_posterior /= np.trapz(fX_posterior, x)

    axes.clear()
    axes.grid(True)

    axes.bar(x, fX_initial, label='$X_{%d}$ initial' % (
        m - 1), edgecolor='black', width=dx)

    axes.bar(x, fX_prior, label='$X_{%d}^{-}$ prior' %
             m, edgecolor='black', width=dx)

    axes.bar(x, fX_posterior,
             label='$X_{%d}^{+}$ posterior' % m, edgecolor='black', width=dx)

    axes.set_xlim(-1, 3)
    axes.set_ylim(0, 4)
    axes.set_xlabel('Position')
    axes.set_ylabel('Prob. density')
    axes.legend()


def hf_demo2():

    fig, axes = subplots(1, figsize=(10, 5))
    show()

    interact(hf_demo2_plot, axes=fixed(axes), M=(10, 200, 10),
             v=(1.0, 4.0, 0.2),
             distX0=distributions,
             sigmaX0=(0.1, 1, 0.1),
             sigmaV=(0.1, 1, 0.1),
             sigmaW=(0.1, 1, 0.1),
             seed=(1, 100, 1))
