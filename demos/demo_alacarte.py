#! /usr/bin/env python3
""" A La Carte GP and basis function demo. """

import logging
import numpy as np
import matplotlib.pyplot as pl
from pyalacarte import basis_functions, regression
from pyalacarte.validation import mll, smse
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
    nbases = 300
    lenscale = 1  # For all basis functions that take lengthscales
    lenscale2 = 0.5  # For the Combo basis
    noise = 1
    order = 5  # For polynomial basis
    rate = 0.9
    eta = 1e-5
    passes = 1000
    batchsize = 100
    reg = 1
    usegradients = True
    diagcov = False

    N = 500
    Ns = 250

    # Dataset selection
    # dataset = 'sinusoid'
    dataset = 'gp1D'

    # Dataset properties
    lenscale_true = 0.7  # For the gpdraw dataset
    noise_true = 0.1

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
        Xtrain = np.linspace(-2 * np.pi, 2 * np.pi, N)[:, np.newaxis]
        ytrain = np.sin(Xtrain).flatten() + np.random.randn(N) * noise
        Xtest = np.linspace(-2 * np.pi, 2 * np.pi, Ns)[:, np.newaxis]
        ftest = np.sin(Xtest).flatten()

    # Random RBF GP
    elif dataset == 'gp1D':

        X = np.linspace(-2 * np.pi, 2 * np.pi, N)
        Xtrain = X[:, np.newaxis]
        Xtest = np.linspace(-2 * np.pi, 2 * np.pi, Ns)[:, np.newaxis]
        Xcat = np.vstack((Xtrain, Xtest))

        K = np.exp(-cdist(Xcat, Xcat, metric='sqeuclidean') /
                   (2 * lenscale_true**2))
        U, S, V = np.linalg.svd(K)
        L = U.dot(np.diag(np.sqrt(S))).dot(V)
        f = np.random.randn(N + Ns).dot(L)

        ytrain = f[0:N] + np.random.randn(N) * noise_true
        ftest = f[N:]

    else:
        raise ValueError('Invalid dataset!')

    #
    # Make Bases
    #

    if basis == 'FF':
        base = basis_functions.FastFood(nbases, Xtrain.shape[1])
    elif basis == 'RKS':
        base = basis_functions.RandomRBF(nbases, Xtrain.shape[1])
    elif basis == 'RBF':
        base = basis_functions.RadialBasis(Xtrain)
    elif basis == 'Linear':
        base = basis_functions.LinearBasis(onescol=True)
    elif basis == 'Poly':
        base = basis_functions.PolynomialBasis(order)
    elif basis == 'Combo':
        base1 = basis_functions.RandomRBF(nbases, Xtrain.shape[1])
        base2 = basis_functions.LinearBasis(onescol=True)
        base3 = basis_functions.FastFood(nbases, Xtrain.shape[1])
        base = base1 + base2 + base3
    else:
        raise ValueError('Invalid basis!')

    #
    # Make real GP
    #

    kdef = lambda h, k: h(1e-5, 1e2, 1) * k('gaussian', h(1e-5, 1e5, lenscale))
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

    # Evidence lower-bound A la Carte learning
    params_sgd = regression.bayeslinear_sgd(Xtrain, ytrain, base, hypers,
                                            var=noise**2, rate=rate, eta=eta,
                                            passes=passes, regulariser=reg,
                                            batchsize=batchsize)
    Ey_s, Vf_s, Vy_s = regression.bayeslinear_predict(Xtest, base, *params_sgd)
    Sy_s = np.sqrt(Vy_s)

    params_elbo = regression.bayeslinear(Xtrain, ytrain, base, hypers,
                                         diagcov=diagcov,
                                         usegradients=usegradients,
                                         regulariser=reg, var=noise**2)
    Ey_e, Vf_e, Vy_e = regression.bayeslinear_predict(Xtest, base,
                                                      *params_elbo)
    Sy_e = np.sqrt(Vy_e)

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

    LL_sgd = mll(ftest, Ey_s, Vf_s)
    LL_elbo = mll(ftest, Ey_e, Vf_e)
    LL_gp = mll(ftest, Ey_gp, Vf_gp)
    smse_sgd = smse(ftest, Ey_s)
    smse_elbo = smse(ftest, Ey_e)
    smse_gp = smse(ftest, Ey_gp)

    log.info("A la Carte (SGD), LL: {}, smse = {}, noise: {}, hypers: {}"
             .format(LL_sgd, smse_sgd, np.sqrt(params_sgd[3]), params_sgd[2]))
    log.info("A la Cart, LL: {}, smse = {}, noise: {}, hypers: {}"
             .format(LL_elbo, smse_elbo, np.sqrt(params_elbo[3]),
                     params_elbo[2]))
    log.info("GP, LL: {}, smse = {}, noise: {}, hypers: {}"
             .format(LL_gp, smse_gp, hyper_params[1], hyper_params[0]))

    #
    # Plot
    #

    Xpl_t = Xtrain.flatten()
    Xpl_s = Xtest.flatten()

    # Training/Truth
    pl.plot(Xpl_t, ytrain, 'k.', label='Training')
    pl.plot(Xpl_s, ftest, 'k-', label='Truth')

    # SGD Regressor
    pl.plot(Xpl_s, Ey_s, 'r-', label='SGD Bayes linear reg.')
    pl.fill_between(Xpl_s, Ey_s - 2 * Sy_s, Ey_s + 2 * Sy_s, facecolor='none',
                    edgecolor='r', linestyle='--', label=None)

    # ELBO Regressor
    pl.plot(Xpl_s, Ey_e, 'g-', label='Bayes linear reg')
    pl.fill_between(Xpl_s, Ey_e - 2 * Sy_e, Ey_e + 2 * Sy_e, facecolor='none',
                    edgecolor='g', linestyle='--', label=None)

    # GP
    pl.plot(Xpl_s, Ey_gp, 'b-', label='GP')
    pl.fill_between(Xpl_s, Ey_gp - 2 * Sy_gp, Ey_gp + 2 * Sy_gp,
                    facecolor='none', edgecolor='b', linestyle='--',
                    label=None)

    pl.legend()

    pl.grid(True)
    pl.title('Regression demo')
    pl.ylabel('y')
    pl.xlabel('x')
    pl.show()


if __name__ == "__main__":
    main()
