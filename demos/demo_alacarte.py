#! /usr/bin/env python3
""" A La Carte GP and basis function demo. """

import logging
import numpy as np
import matplotlib.pyplot as pl
from pyalacarte import bases, regression
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
    lenscale = 0.5  # For all basis functions that take lengthscales
    lenscale2 = 0.2  # For the Combo basis
    noise = 0.2
    order = 5  # For polynomial basis
    rate = 0.9
    eta = 1e-5
    passes = 1000
    batchsize = 100
    reg = 1
    usegradients = True
    useSGD = True
    diagcov = True

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
        base = bases.FastFood(nbases, Xtrain.shape[1])
    elif basis == 'RKS':
        base = bases.RandomRBF(nbases, Xtrain.shape[1])
    elif basis == 'RBF':
        base = bases.RadialBasis(Xtrain)
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

    # Log marginal likelihood A La Carte learning
    params_lml = regression.bayesreg_lml(Xtrain, ytrain, base, hypers,
                                         usegradients=usegradients,
                                         regulariser=reg, var=noise**2)
    Ey_l, Vf_l, Vy_l = regression.bayesreg_predict(Xtest, base, *params_lml)
    Sy_l = np.sqrt(Vy_l)

    # Evidence lower-bound A la Carte learning
    if useSGD:
        params_elbo = regression.bayesreg_sgd(Xtrain, ytrain, base, hypers,
                                              var=noise**2, rate=rate, eta=eta,
                                              passes=passes, regulariser=reg,
                                              batchsize=batchsize)
    else:
        params_elbo = regression.bayesreg_elbo(Xtrain, ytrain, base, hypers,
                                               diagcov=diagcov,
                                               usegradients=usegradients,
                                               regulariser=reg, var=noise**2)
    Ey_e, Vf_e, Vy_e = regression.bayesreg_predict(Xtest, base, *params_elbo)
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

    LL_lml = mll(ftest, Ey_l, Vf_l)
    LL_elbo = mll(ftest, Ey_e, Vf_e)
    LL_gp = mll(ftest, Ey_gp, Vf_gp)
    smse_lml = smse(ftest, Ey_l)
    smse_elbo = smse(ftest, Ey_e)
    smse_gp = smse(ftest, Ey_gp)

    log.info("A la Carte (LML), LL: {}, smse = {}, noise: {}, hypers: {}"
             .format(LL_lml, smse_lml, np.sqrt(params_lml[3]), params_lml[2]))
    log.info("A la Carte (ELBO), LL: {}, smse = {}, noise: {}, hypers: {}"
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
    pl.plot(Xpl_t, ytrain, 'k.', Xpl_s, ftest, 'k-')

    # LML Regressor
    pl.plot(Xpl_s, Ey_l, 'r-')
    pl.fill_between(Xpl_s, Ey_l - 2 * Sy_l, Ey_l + 2 * Sy_l, facecolor='none',
                    edgecolor='r', linestyle='--', label=None)

    # ELBO Regressor
    pl.plot(Xpl_s, Ey_e, 'g-')
    pl.fill_between(Xpl_s, Ey_e - 2 * Sy_e, Ey_e + 2 * Sy_e, facecolor='none',
                    edgecolor='g', linestyle='--', label=None)

    # GP
    pl.plot(Xpl_s, Ey_gp, 'b-')
    pl.fill_between(Xpl_s, Ey_gp - 2 * Sy_gp, Ey_gp + 2 * Sy_gp,
                    facecolor='none', edgecolor='b', linestyle='--',
                    label=None)

    pl.legend(['Training', 'Truth', 'A la Carte (LML)', 'A la Carte (ELBO)',
               'GP'])

    pl.show()


if __name__ == "__main__":
    main()
