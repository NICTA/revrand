#! /usr/bin/env python
""" A La Carte GP and basis function demo. """

import matplotlib.pyplot as pl
import numpy as np
import logging

from revrand import slm, glm, likelihoods
from revrand.metrics import mll, smse
from revrand.utils.datasets import gen_gausprocess_se
from revrand.btypes import Parameter, Positive
from revrand import basis_functions as bs
import revrand.legacygp as gp
import revrand.legacygp.kernels as kern

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():

    #
    # Settings
    #

    # Dataset properties
    N = 300
    Ns = 250

    # Dataset selection
    # dataset = 'sinusoid'
    dataset = 'gp1D'
    lenscale_true = 1.2  # For the gpdraw dataset
    noise_true = 0.1

    # Algorithmic properties
    nbases = 100
    lenscale = 1  # For all basis functions that take lengthscales
    noise = 0.5
    reg = 1

    # GLM learning settings
    maxiter = 3000
    batch_size = 10

    lenp = Parameter(lenscale, Positive())
    base = bs.RandomRBF(Xdim=1, nbases=nbases, lenscale_init=lenp)
    # base = bs.RandomMatern32(Xdim=1, nbases=nbases, lenscale_init=lenp)

    # Gaussian spectral mixture basis
    # ncomponents = 5
    # basedpercomp = int(np.round(nbases / (ncomponents * 4)))
    # base = bs.spectralmixture(Xdim=1, ncomponents=ncomponents,
                              # bases_per_component=basedpercomp,
                              # lenscales_init=[lenp] * ncomponents)

    #
    # Make Data
    #

    # Sinusoid
    if dataset == 'sinusoid':
        Xtrain = np.random.rand(N)[:, np.newaxis] * 4 * np.pi - 2 * np.pi
        ytrain = np.sin(Xtrain).flatten() + np.random.randn(N) * noise_true
        Xtest = np.linspace(-2 * np.pi, 2 * np.pi, Ns)[:, np.newaxis]
        ftest = np.sin(Xtest).flatten()

    # Random RBF GP
    elif dataset == 'gp1D':

        Xtrain, ytrain, Xtest, ftest = \
            gen_gausprocess_se(N, Ns, lenscale=lenscale_true, noise=noise_true)

    else:
        raise ValueError('Invalid dataset!')

    #
    # Learn regression parameters and predict
    #

    params_slm = slm.learn(Xtrain, ytrain, base,
                           var=Parameter(noise**2, Positive()),
                           regulariser=Parameter(reg, Positive()))
    Ey_e, Vf_e, Vy_e = slm.predict(Xtest, base, *params_slm)
    Sy_e = np.sqrt(Vy_e)

    #
    # Nonparametric variational inference GLM
    #

    llhood = likelihoods.Gaussian(var_init=Parameter(noise**2, Positive()))
    params_glm = glm.learn(Xtrain, ytrain, llhood, base,
                           regulariser=Parameter(reg, Positive()),
                           use_sgd=True, batch_size=batch_size,
                           maxiter=maxiter)
    Ey_g, Vf_g, Eyn, Eyx = glm.predict_moments(Xtest, llhood, base,
                                               *params_glm)
    Vy_g = Vf_g + params_glm[2][0]
    Sy_g = np.sqrt(Vy_g)

    #
    # Learn GP and predict
    #

    def kdef(h, k):
        return (h(1e-5, 1., 0.5) * k(kern.gaussian, h(1e-5, 1e5, lenscale)) +
                k(kern.lognoise, h(-4, 1, -3)))
    hyper_params = gp.learn(Xtrain, ytrain, kdef, verbose=True, ftol=1e-6,
                            maxiter=maxiter)

    regressor = gp.condition(Xtrain, ytrain, kdef, hyper_params)
    query = gp.query(regressor, Xtest)
    Ey_gp = gp.mean(query)
    Vy_gp = gp.variance(query, noise=True)
    Sy_gp = np.sqrt(Vy_gp)

    #
    # Evaluate LL and SMSE
    #

    LL_s = mll(ftest, Ey_e, Vy_e)
    LL_gp = mll(ftest, Ey_gp, Vy_gp)
    LL_g = mll(ftest, Ey_g, Vy_g)

    smse_s = smse(ftest, Ey_e)
    smse_gp = smse(ftest, Ey_gp)
    smse_glm = smse(ftest, Ey_g)

    log.info("SLM, LL: {}, smse = {}, noise: {}, hypers: {}"
             .format(LL_s, smse_s, np.sqrt(params_slm[3]),
                     params_slm[2]))
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
    pl.plot(Xpl_s, Ey_gp, 'b-', label='GP')
    pl.fill_between(Xpl_s, Ey_gp - 2 * Sy_gp, Ey_gp + 2 * Sy_gp,
                    facecolor='none', edgecolor='b', linestyle='--',
                    label=None)

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
