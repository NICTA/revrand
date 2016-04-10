#! /usr/bin/env python3
""" A La Carte GP and basis function demo. """

import matplotlib.pyplot as pl
import revrand.legacygp as gp
import revrand.legacygp.kernels as kern
import numpy as np
import logging

from revrand import basis_functions, regression, glm, likelihoods
from revrand.validation import mll, smse
from revrand.utils.datasets import gen_gausprocess_se

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():

    #
    # Settings
    #

    # Algorithmic properties
    nbases = 50
    lenscale = 1  # For all basis functions that take lengthscales
    lenscale2 = 0.5  # For the Combo basis
    noise = 1
    order = 7  # For polynomial basis
    rate = 0.9
    eta = 1e-5
    passes = 1000
    batchsize = 100
    reg = 1

    # np.random.seed(100)

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

        Xtrain, ytrain, Xtest, ftest = \
            gen_gausprocess_se(N, Ns, lenscale=lenscale_true, noise=noise_true)

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

    # Set up optimisation
    # learning_params = gp.OptConfig()
    # learning_params.sigma = gp.auto_range(kdef)
    # learning_params.noise = gp.Range([1e-5], [1e5], [1])
    # learning_params.walltime = 60

    #
    # Learn regression parameters and predict
    #

    if basis == 'Linear' or basis == 'Poly':
        hypers = []
    elif basis == 'FF' or basis == 'RKS' or basis == 'RBF':
        hypers = [lenscale]
    elif basis == 'Combo':
        hypers = [lenscale, lenscale2]
    else:
        raise ValueError('Invalid basis!')

    params_elbo = regression.learn(Xtrain, ytrain, base, hypers, var=noise**2,
                                   regulariser=reg)
    Ey_e, Vf_e, Vy_e = regression.predict(Xtest, base, *params_elbo)
    Sy_e = np.sqrt(Vy_e)

    #
    # Nonparametric variational inference GLM
    #

    llhood = likelihoods.Gaussian()
    lparams = [noise**2]
    params_glm = glm.learn(Xtrain, ytrain, llhood, lparams, base, hypers,
                           regulariser=reg, use_sgd=True, rate=rate,
                           postcomp=10, eta=eta, batchsize=batchsize,
                           maxit=passes)
    Ey_g, Vf_g, Eyn, Eyx = glm.predict_meanvar(Xtest, llhood, base,
                                               *params_glm)
    Vy_g = Vf_g + params_glm[2][0]
    Sy_g = np.sqrt(Vy_g)

    #
    # Learn GP and predict
    #

    def kdef(h, k):
        return (h(1e-5, 1., 0.5) * k(kern.gaussian, h(1e-5, 1e5, lenscale)) +
                k(kern.lognoise, h(-4, 1, -3)))
    hyper_params = gp.learn(Xtrain, ytrain, kdef, verbose=True, ftol=1e-15,
                            maxiter=passes)

    regressor = gp.condition(Xtrain, ytrain, kdef, hyper_params)
    query = gp.query(regressor, Xtest)
    Ey_gp = gp.mean(query)
    Vf_gp = gp.variance(query)
    Vy_gp = gp.variance(query, noise=True)
    Sy_gp = np.sqrt(Vy_gp)

    # import ipdb; ipdb.set_trace()

    #
    # Evaluate LL and SMSE
    #

    LL_elbo = mll(ftest, Ey_e, Vf_e)
    LL_gp = mll(ftest, Ey_gp, Vf_gp)
    LL_g = mll(ftest, Ey_g, Vy_g)

    smse_elbo = smse(ftest, Ey_e)
    smse_gp = smse(ftest, Ey_gp)
    smse_glm = smse(ftest, Ey_g)

    log.info("A la Carte, LL: {}, smse = {}, noise: {}, hypers: {}"
             .format(LL_elbo, smse_elbo, np.sqrt(params_elbo[3]),
                     params_elbo[2]))
    log.info("GP, LL: {}, smse = {}, noise: {}, hypers: {}"
             .format(LL_gp, smse_gp, hyper_params[1], hyper_params[0]))
    log.info("GLM, LL: {}, smse = {}, noise: {}, hypers: {}"
             .format(LL_g, smse_glm, np.sqrt(params_glm[2][0]),
                     params_glm[3]))

    #
    # Plot
    #

    Xpl_t = Xtrain.flatten()
    Xpl_s = Xtest.flatten()

    # Training/Truth
    pl.plot(Xpl_t, ytrain, 'k.', label='Training')
    pl.plot(Xpl_s, ftest, 'k-', label='Truth')

    # ELBO Regressor
    pl.plot(Xpl_s, Ey_e, 'g-', label='Bayesian linear regression')
    pl.fill_between(Xpl_s, Ey_e - 2 * Sy_e, Ey_e + 2 * Sy_e, facecolor='none',
                    edgecolor='g', linestyle='--', label=None)

    # GP
    # pl.plot(Xpl_s, Ey_gp, 'b-', label='GP')
    # pl.fill_between(Xpl_s, Ey_gp - 2 * Sy_gp, Ey_gp + 2 * Sy_gp,
    #                 facecolor='none', edgecolor='b', linestyle='--',
    #                 label=None)

    # GLM Regressor
    pl.plot(Xpl_s, Ey_g, 'm-', label='GLM')
    pl.fill_between(Xpl_s, Ey_g - 2 * Sy_g, Ey_g + 2 * Sy_g, facecolor='none',
                    edgecolor='m', linestyle='--', label=None)

    pl.legend()

    pl.grid(True)
    pl.title('Regression demo')
    pl.ylabel('y')
    pl.xlabel('x')

    pl.show()


if __name__ == "__main__":
    main()
