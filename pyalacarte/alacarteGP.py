""" A simple implementation of the "A la Carte" GP [1].

    Authors:    Daniel Steinberg, Lachlan McCalman
    Date:       8 May 2015
    Institute:  NICTA

    References:
    [1] Yang, Z., Smola, A. J., Song, L., & Wilson, A. G. "A la Carte --
        Learning Fast Kernels". Proceedings of the Eighteenth International
        Conference on Artificial Intelligence and Statistics, pp. 1098–1106,
        2015.
"""

import numpy as np
import logging
from scipy.linalg import cho_solve
from pyalacarte.linalg import jitchol, logdet
from pyalacarte.minimize import minimize as nmin


# Set up logging
log = logging.getLogger(__name__)


def alacarte_learn(X, y, basis, hypers, noise=1, regulariser=1e-3, ftol=1e-5,
                   maxit=1000, verbose=True, noiseLB=1e-7, regulariserLB=1e-7,
                   usegradients=True):
    """ Learn the parameters and hyperparameters of an "A la Carte" [1]
        Gaussian Process.

        Arguments:
            X: NxD array input dataset (N samples, D dimensions)
            y: N array targets (N samples)
            basis: A basis object, see bases.py
            hypers: A sequence of hyperparameters of the basis object
            noise: observation noise initial guess
            regulariser: weight regulariser (variance) initial guess
            verbose: log learning status
            ftol: optimiser function tolerance convergence criterion
            maxit: maximum number of iterations for the optimiser
            noiseLB: lower bound on the noise parameter
            regulariserLB: lower bound on the regulariser
            usegradients: True for using gradients to optimize the parameters,
                otherwize false uses BOBYQA (from nlopt)

        Returns:
            hypers: learned sequence of basis object hyperparameters
            noise: learned observation noise
            regulariser: learned weight regluariser
    """

    N, d = X.shape
    var = noise**2

    def LML(params):

        _var = params[0]
        _lambda = params[1]
        _theta = np.atleast_1d(params[2:])
        grad = np.empty(len(params))

        # Common computations
        Phi = basis.get_basis(X, *_theta)                       # N x D
        PhiPhi = Phi.T.dot(Phi)                                 # D x D
        D = Phi.shape[1]
        LC = jitchol(np.diag(np.ones(D) / _lambda) + PhiPhi / _var)
        iCPhi = cho_solve(LC, Phi.T)                            # D x N
        yiK = y.T / _var - (y.T.dot(Phi)).dot(iCPhi) / _var**2  # 1 x N

        LML = - 0.5 * (N * np.log(2*np.pi*_var**2) + D * np.log(_lambda)
                       + logdet(LC[0]) + yiK.dot(y))

        if verbose:
            log.info("LML = {}, noise = {}, reg = {}, hypers = {}."
                     .format(LML, np.sqrt(_var), _lambda, _theta))

        if not usegradients:
            return -LML

        # Grad var
        grad[0] = - N / _var + (Phi * iCPhi.T).sum() / (2 * _var**2) \
            + (yiK**2).sum() / 2

        # Grad reg -- the second trace here is the largest computation
        grad[1] = 0.5 * (- np.trace(PhiPhi) / _var
                         + (PhiPhi * (iCPhi).dot(Phi)).sum() / _var**2
                         + (yiK.dot(Phi)**2).sum())

        # Loop through basis param grads
        dPhis = basis.get_grad(X, *_theta)  # if one of these is empty, skip
        for i, (t, dPhi) in enumerate(zip(_theta, dPhis)):
            dPhiPhi = dPhi.T.dot(Phi)  # D x D
            gt = - (np.trace(dPhiPhi) / _var
                    - (dPhiPhi * (iCPhi.dot(Phi))).sum() / _var**2  # expensive
                    + (yiK.dot(dPhi)).dot(Phi.T).dot(yiK.T)) * _lambda
            grad[2+i] = gt

        return -LML, -grad

    params = [var, regulariser] + list(hypers)
    bounds = [(noiseLB**2, None), (regulariserLB, None)] + basis.get_bounds()

    method = 'L-BFGS-B' if usegradients else None  # else BOBYQA numerical
    res = nmin(LML, params, bounds=bounds, method=method, ftol=ftol, xtol=1e-8,
               maxiter=maxit)

    noise = np.sqrt(res['x'][0])
    regulariser = res['x'][1]
    hypers = np.atleast_1d(res['x'][2:])

    if verbose:
        log.info("Done! LML = {}, noise = {}, reg = {}, hypers = {}."
                 .format(-res['fun'], noise, regulariser, hypers))
        if not res['success']:
            log.info('Terminated unsuccessfully: {}.'.format(res['message']))

    return hypers, noise, regulariser


def alacarte_predict(X_star, X, y, basis, hypers, noise, regulariser):
    """ Predict using an "A la Carte" Gaussian process [1].

        Arguments:
            X_star: N_starxD array query input dataset (N_star samples,
                D dimensions)
            X: NxD array training input dataset (N samples, D dimensions)
            y: N array training targets (N samples)
            basis: A basis object, see bases.py
            hypers: A sequence of hyperparameters of the basis object
            noise: observation noise initial guess
            regulariser: weight regulariser (variance) initial guess

        Returns:
            Two N_star lenght arrays,
            Ey: The expected value of y_star for the query inputs, X_star
            Vf: The expected variance of f_star for the query inputs, X_star
            Vy: The expected variance of y_star for the query inputs, X_star
    """

    var = noise**2
    Phi = basis.get_basis(X, *hypers)
    Phi_s = basis.get_basis(X_star, *hypers)
    N, D = Phi.shape

    LC = jitchol(np.diag(np.ones(D) / regulariser) + Phi.T.dot(Phi) / var)

    Ey = Phi_s.dot(cho_solve(LC, Phi.T.dot(y))) / var
    Vf = (Phi_s * cho_solve(LC, Phi_s.T).T).sum(axis=1)

    return Ey, Vf, Vf + var
