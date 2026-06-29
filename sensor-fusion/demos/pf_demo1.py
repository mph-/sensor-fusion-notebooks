# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed
from scipy.interpolate import interp1d
from matplotlib.pyplot import subplots, show
from .lib.utils import gauss
from .lib.kde import KDE as KDE1

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


def pf_demo1_plot(axes, distX0='gaussian', sigmaX0=0.1, sigmaV=0.1,
                  sigmaW=0.05, seed=1, num_particles=5, z=1,
                  resample=False, KDE=False, annotate=True):

    np.random.seed(seed)

    step = 1
    v = 1

    dt = 1
    B = dt

    annotate_max = 5

    Nx = 1000
    x = np.linspace(-10, 40, Nx)

    fX = pdf(x, 0, sigmaX0, distX0)

    px_initial = sample(x, fX, num_particles)
    weights_initial = np.ones(num_particles)

    px_posterior = 0
    weights_posterior = 0

    for m in range(1, step + 1):

        if m > 1:
            px_initial = px_posterior
            weights_initial = weights_posterior

        px_prior = px_initial + B * v + np.random.randn(num_particles) * sigmaW
        weights_prior = weights_initial

        # z = C * m * dt * v + np.random.randn(1) * sigmaV

        px_posterior = px_prior
        weights_posterior = weights_prior * gauss(px_posterior, z, sigmaV)
        weights_posterior1 = weights_posterior

        if resample:
            fXpostest = KDE1(px_posterior, weights_posterior).estimate(x)
            px_posterior = sample(x, fXpostest, num_particles)
            weights_posterior = np.ones(num_particles)

    axes.clear()
    axes.grid(True)

    width = 0.02
    alpha = 0.5
    axes.bar(px_initial, weights_initial, width=width, linewidth=0,
             alpha=alpha, label='$X_{%d}$ initial' % (m - 1))

    idx = np.argsort(px_initial[0: min(num_particles, annotate_max)])

    if annotate:
        for q, p in enumerate(idx):
            axes.text(px_initial[p], weights_initial[p] * 1.05, '%d' % (q + 1))

    axes.bar(px_prior, weights_prior, width=width, linewidth=0,
             alpha=alpha, label='$X_{%d}^{-}$ prior' % m)
    if annotate:
        for q, p in enumerate(idx):
            axes.text(px_prior[p], weights_prior[p] * 1.05, '%d' % (q + 1))

    axes.bar(px_posterior, weights_posterior, width=width, linewidth=0,
             alpha=alpha, label='$X_{%d}^{+}$ posterior' % m)

    max_weight = max(max(weights_initial), max(weights_prior),
                     max(weights_posterior1))
    _ = max_weight

    axes.set_xlim(-1, 2)
    # axes.set_ylim(0, max_weight + 0.1)
    axes.set_ylim(0, 4.1)
    axes.set_xlabel('Position')
    axes.set_ylabel('Weight')

    if KDE:
        fXpostest = KDE1(px_posterior, weights_posterior).estimate(x)
        fXpriortest = KDE1(px_prior, weights_prior).estimate(x)
        fXinitialest = KDE1(px_initial, weights_initial).estimate(x)

        ax2 = axes.twinx()
        ax2.plot(x, fXinitialest)
        ax2.plot(x, fXpriortest)
        ax2.plot(x, fXpostest)
        ax2.set_ylabel('Prob. density')

    axes.legend()


def pf_demo1():

    fig, axes = subplots(1)
    show()

    interact(pf_demo1_plot, axes=fixed(axes),
             step=(1, 5), num_particles=(5, 100, 5),
             z=(0.7, 1.3, 0.05),
             v=(1.0, 4.0, 0.2),
             distX0=distributions,
             sigmaX0=(0, 0.2, 0.02),
             sigmaV=(0.1, 0.2, 0.02),
             sigmaW=(0.0, 0.2, 0.02),
             seed=(1, 100, 1),
             continuous_update=False)
