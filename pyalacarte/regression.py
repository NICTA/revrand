""" A simple implementation of the "A la Carte" GP [1].

    References:
        - Yang, Z., Smola, A. J., Song, L., & Wilson, A. G. "A la Carte --
          Learning Fast Kernels". Proceedings of the Eighteenth International
          Conference on Artificial Intelligence and Statistics, pp. 1098-1106,
          2015.
"""

import numpy as np
import logging
from scipy.linalg import cho_solve
from pyalacarte.linalg import jitchol, logdet
from pyalacarte.minimize import minimize as nmin
from pyalacarte.bases import params_to_list, list_to_params


# Set up logging
log = logging.getLogger(__name__)


def alacarte_learn(X, y, basis, bparams, noise=1, regulariser=1e-3, ftol=1e-5,
                   maxit=1000, verbose=True, noise_bounds=(1e-7, None),
                   regulariser_bounds=(1e-7, None), usegradients=True):
    """ Learn the parameters and hyperparameters of an "A la Carte" Gaussian
        Process.

        Arguments:
            X: NxD array input dataset (N samples, D dimensions)
            y: N array targets (N samples)
            basis: A basis object, see bases.py
            bparams: A sequence of parameters of the basis object
            noise: observation noise initial guess
            regulariser: weight regulariser (variance) initial guess
            verbose: log learning status
            ftol: optimiser function tolerance convergence criterion
            maxit: maximum number of iterations for the optimiser
            noise_bounds: tuple of (lower bound, upper bound) on the noise
                parameter, None for unbounded (though it cannot be <= 0)
            regulariser_bounds: tuple of (lower bound, upper bound) on the
                regulariser parameter, None for unbounded (though it cannot be
                <= 0)
            usegradients: True for using gradients to optimize the parameters,
                otherwize false uses BOBYQA (from nlopt)

        Returns:
            list: learned sequence of basis object hyperparameters
            float: learned observation noise
            float: learned weight regluariser
    """

    N, d = X.shape
    var = noise**2

    def LML(params):

        _var = params[0]
        _lambda = params[1]
        _theta = np.atleast_1d(params[2:]).tolist()

        # Common computations
        Phi = basis.from_vector(X, _theta)                      # N x D
        PhiPhi = Phi.T.dot(Phi)                                 # D x D
        D = Phi.shape[1]
        LC = jitchol(np.diag(np.ones(D) / _lambda) + PhiPhi / _var)
        iCPhi = cho_solve(LC, Phi.T)                            # D x N
        yiK = y.T / _var - (y.T.dot(Phi)).dot(iCPhi) / _var**2  # 1 x N

        # Objective
        LML = - 0.5 * (N * np.log(2*np.pi*_var**2) + D * np.log(_lambda)
                       + logdet(LC[0]) + yiK.dot(y))

        if verbose:
            log.info("LML = {}, noise = {}, reg = {}, bparams = {}."
                     .format(LML, np.sqrt(_var), _lambda, _theta))

        if not usegradients:
            return -LML

        # Gradients
        grad = np.empty(len(params))

        # Grad var
        grad[0] = - N / _var + (Phi * iCPhi.T).sum() / (2 * _var**2) \
            + (yiK**2).sum() / 2

        # Grad reg -- the second trace here is the largest computation
        grad[1] = 0.5 * (- np.trace(PhiPhi) / _var
                         + (PhiPhi * (iCPhi).dot(Phi)).sum() / _var**2
                         + (yiK.dot(Phi)**2).sum())

        # Loop through basis param grads
        dPhis = basis.grad_from_vector(X, _theta) if _theta else []
        for i,  dPhi in enumerate(dPhis):
            dPhiPhi = dPhi.T.dot(Phi)  # D x D
            grad[2+i] = - (np.trace(dPhiPhi) / _var
                           - (dPhiPhi * (iCPhi.dot(Phi))).sum() / _var**2  # !
                           + (yiK.dot(dPhi)).dot(Phi.T).dot(yiK.T)) * _lambda

        return -LML, -grad

    params = [var, regulariser] + params_to_list(bparams)
    bounds = [(noise_bounds[0]**2, noise_bounds[1]), regulariser_bounds] \
        + basis.bounds

    method = 'L-BFGS-B' if usegradients else None  # else BOBYQA numerical
    res = nmin(LML, params, bounds=bounds, method=method, ftol=ftol, xtol=1e-8,
               maxiter=maxit)

    noise = np.sqrt(res['x'][0])
    regulariser = res['x'][1]
    bparams = list_to_params(bparams, np.atleast_1d(res['x'][2:]))

    if verbose:
        log.info("Done! LML = {}, noise = {}, reg = {}, bparams = {}."
                 .format(-res['fun'], noise, regulariser, bparams))
        if not res['success']:
            log.info('Terminated unsuccessfully: {}.'.format(res['message']))

    return bparams, noise, regulariser


def alacarte_predict(X_star, X, y, basis, bparams, noise, regulariser):
    """ Predict using an "A la Carte" Gaussian process [1].

        Arguments:
            X_star: N_starxD array query input dataset (N_star samples,
                D dimensions)
            X: NxD array training input dataset (N samples, D dimensions)
            y: N array training targets (N samples)
            basis: A basis object, see bases.py
            bparams: A sequence of hyperparameters of the basis object
            noise: observation noise initial guess
            regulariser: weight regulariser (variance) initial guess

        Returns:
            array: The expected value of y_star for the query inputs, X_star
               of shape (N_star,)
            array: The expected variance of f_star for the query inputs, X_star
               of shape (N_star,)
            array: The expected variance of y_star for the query inputs, X_star
               of shape (N_star,)
    """

    var = noise**2
    Phi = basis(X, *bparams)
    Phi_s = basis(X_star, *bparams)
    N, D = Phi.shape

    LC = jitchol(np.diag(np.ones(D) / regulariser) + Phi.T.dot(Phi) / var)

    Ey = Phi_s.dot(cho_solve(LC, Phi.T.dot(y))) / var
    Vf = (Phi_s * cho_solve(LC, Phi_s.T).T).sum(axis=1)

    return Ey, Vf, Vf + var
