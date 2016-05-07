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

        m, C, bparams, var = slm.learn(X, y,
                                       basis=self.basis,
                                       var=self.var,
                                       regulariser=self.regulariser,
                                       tol=self.tol,
                                       maxit=self.maxit,
                                       verbose=self.verbose
                                       )
        self.m = m
        self.C = C
        self.bparams = bparams
        self.optvar = var

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
