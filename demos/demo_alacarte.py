#! /usr/bin/env python3
""" A La Carte GP and basis function demo. """

import logging
import numpy as np
import matplotlib.pyplot as pl
from pyalacarte import bases, alacarteGP
from pyalacarte.validation import mll
from scipy.spatial.distance import cdist
import computers.gp as gp

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():

    #
    # Settings
    #

    # Algorithmic properties
    nbases = 500
    lenscale = 0.1  # For all basis functions that take lengthscales
    lenscale2 = 0.2  # For the Combo basis
    order = 5  # For polynomial basis
    usegradients = True

    N = 5e2
    Ns = 1e3

    # Dataset selection
    #dataset = 'sinusoid'
    dataset = 'gp1D'

    # Dataset properties
    lenscale_true = 0.7  # For the gpdraw dataset
    noise = 0.1

    basis = 'RKS'
    # basis = 'FF'
    # basis = 'RBF'
    # basis = 'Linear'
    # basis = 'Poly'
    # basis = 'Combo'

    #
    # Make Data
    #

    # Sinusoid
    if dataset == 'sinusoid':
        Xtrain = np.linspace(-2*np.pi, 2*np.pi, N)[:, np.newaxis]
        ytrain = np.sin(Xtrain).flatten() + np.random.randn(N) * noise
        Xtest = np.linspace(-2*np.pi, 2*np.pi, Ns)[:, np.newaxis]
        ftest = np.sin(Xtest).flatten()

    # Random RBF GP
    elif dataset == 'gp1D':

        X = np.linspace(-2*np.pi, 2*np.pi, N)
        Xtrain = X[:, np.newaxis]
        Xtest = np.linspace(-2*np.pi, 2*np.pi, Ns)[:, np.newaxis]
        Xcat = np.vstack((Xtrain, Xtest))

        K = np.exp(-cdist(Xcat, Xcat, metric='sqeuclidean')
                   / (2*lenscale_true**2))
        U, S, V = np.linalg.svd(K)
        L = U.dot(np.diag(np.sqrt(S))).dot(V)
        f = np.random.randn(N + Ns).dot(L)

        ytrain = f[0:N] + np.random.randn(N) * noise
        ftest = f[N:]

    else:
        raise ValueError('Invalid dataset!')

    #
    # Make Bases
    #

    if basis == 'FF':
        base = bases.FastFood(nbases, Xtrain.shape[1])
    elif basis == 'RKS':
        base = bases.RandomRBF(nbases, Xtrain.shape[1])
    elif basis == 'RBF':
        base = bases.RadialBasis(Xtrain, onescol=False)
    elif basis == 'Linear':
        base = bases.LinearBasis(onescol=True)
    elif basis == 'Poly':
        base = bases.PolynomialBasis(order)
    elif basis == 'Combo':
        base1 = bases.RandomRBF(nbases, Xtrain.shape[1])
        base2 = bases.LinearBasis(onescol=True)
        base3 = bases.FastFood(nbases, Xtrain.shape[1])
        base = base1 + base2 + base3
    else:
        raise ValueError('Invalid basis!')

    #
    # Make real GP
    #

    kdef = lambda h, k: h(1e-5, 1e2, 1)*k('gaussian', h(1e-5, 1e5, lenscale))
    kfunc = gp.compose(kdef)

    # Set up optimisation
    learning_params = gp.OptConfig()
    learning_params.sigma = gp.auto_range(kdef)
    learning_params.noise = gp.Range([1e-5], [1e5], [1])
    learning_params.walltime = 60

    #
    # Learn regression parameters and predict
    #

    if basis == 'Linear' or basis == 'Poly':
        hypers = ()
    elif basis == 'FF' or basis == 'RKS' or basis == 'RBF':
        hypers = (lenscale,)
    elif basis == 'Combo':
        hypers = (lenscale, lenscale2)
    else:
        raise ValueError('Invalid basis!')

    params = alacarteGP.alacarte_learn(Xtrain, ytrain, base, hypers,
                                       usegradients=usegradients)
    Ey, Vf, Vy = alacarteGP.alacarte_predict(Xtest, Xtrain, ytrain, base,
                                             *params)
    Sy = np.sqrt(Vy)

    #
    # Learn GP and predict
    #

    hyper_params = gp.learn(Xtrain, ytrain, kfunc, learning_params)
    regressor = gp.condition(Xtrain, ytrain, kfunc, hyper_params)

    query = gp.query(Xtest, regressor)
    Ey_gp = gp.mean(regressor, query)
    Vf_gp = gp.variance(regressor, query)
    Vy_gp = Vf_gp + np.array(hyper_params[1])**2
    Sy_gp = np.sqrt(Vy_gp)

    #
    # Evaluate LL
    #

    LL_ala = mll(ftest, Ey, Vf)
    LL_gp = mll(ftest, Ey_gp, Vf_gp)

    log.info("A la Carte LL: {}, noise: {}, hypers: {}"
             .format(LL_ala, params[1], params[0]))
    log.info("GP LL: {}, noise: {}, hypers: {}"
             .format(LL_gp, hyper_params[1], hyper_params[0]))

    #
    # Plot
    #

    Xpl_t = Xtrain.flatten()
    Xpl_s = Xtest.flatten()

    # Regressor
    pl.plot(Xpl_t, ytrain, 'k.', Xpl_s, ftest, 'g-', Xpl_s, Ey, 'r-')
    pl.fill_between(Xpl_s, Ey - 2*Sy, Ey + 2*Sy, facecolor='r', alpha=0.3,
                    edgecolor='none', label=None)

    # GP
    pl.plot(Xpl_s, Ey_gp, 'b-')
    pl.fill_between(Xpl_s, Ey_gp - 2*Sy_gp, Ey_gp + 2*Sy_gp, facecolor='b',
                    alpha=0.3, edgecolor='none', label=None)

    pl.legend(['Training', 'Test', 'A la Carte', 'GP'])
    pl.show()


if __name__ == "__main__":
    main()
