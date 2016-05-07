""" Scikit learn interface -- compatible with pipelines """

from sklearn.base import BaseEstimator

from . import slm, glm
from .btypes import Parameter, Positive


class StandardLinearModel(BaseEstimator):

    def __init__(self, basis, var=Parameter(1., Positive()),
                 regulariser=Parameter(1., Positive()), tol=1e-6, maxit=500,
                 verbose=True):

        self.basis = basis
        self.var = var
        self.regulariser = regulariser
        self.tol = tol
        self.maxit = maxit
        self.verbose = verbose

    def fit(self, X, y):

        self.m, self.C, self.bparams, self.optvar = \
            slm.learn(X, y,
                      basis=self.basis,
                      var=self.var,
                      regulariser=self.regulariser,
                      tol=self.tol,
                      maxit=self.maxit,
                      verbose=self.verbose
                      )

        return self

    def predict(self, X, uncertainty=False):

        Ey, Vf, Vy = slm.predict(X,
                                 self.basis,
                                 self.m,
                                 self.C,
                                 self.bparams,
                                 self.optvar
                                 )

        return (Ey, Vf, Vy) if uncertainty else Ey


class GeneralisedLinearModel(BaseEstimator):

    def __init__(self, likelihood, basis, 
                 regulariser=Parameter(1., Positive()), postcomp=10,
                 use_sgd=True, maxit=1000, tol=1e-7, batchsize=100, rho=0.9,
                 epsilon=1e-5, verbose=True):

        self.likelihood = likelihood
        self.basis = basis
        self.regulariser = regulariser
        self.postcomp = postcomp
        self.use_sgd = use_sgd
        self.maxit = maxit
        self.tol = tol
        self.batchsize = batchsize
        self.rho = rho
        self.epsilon = epsilon
        self.verbose = verbose

    def fit(self, X, y):

        self.m, self.C, self.lparams, self.bparams = \
            glm.learn(X, y,
                      likelihood=self.likelihood,
                      basis=self.basis, 
                      regulariser=self.regulariser,
                      postcomp=self.postcomp,
                      use_sgd=self.use_sgd,
                      maxit=self.maxit,
                      tol=self.tol,
                      batchsize=self.batchsize,
                      rho=self.rho,
                      epsilon=self.epsilon,
                      verbose=self.verbose)

        return self

    def predict(self, X, prediction_type=None, *args, **kwargs):

        predargs = [X,
                    self.likelihood, 
                    self.basis,
                    self.m,
                    self.C,
                    self.lparams,
                    self.bparams]

        Ey, Vy, _, _ = glm.predict_moments(*predargs)

        if prediction_type is None:
            return Ey
        elif prediction_type == 'variance':
            return Ey, Vy
        elif prediction_type == 'interval':
            raise NotImplementedError()
        elif prediction_type == 'cdf':
            raise NotImplementedError()
        else:
            raise ValueError("Invalid prediction type input.")
