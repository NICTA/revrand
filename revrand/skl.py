""" Scikit learn interface -- compatible with pipelines """

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from . import slm, glm
from . import basis_functions as bf
from .btypes import Parameter, Positive


class StandardLinearModel(BaseEstimator, RegressorMixin):

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

        self.m, self.C, self.hypers, self.optvar = \
            slm.learn(X, y,
                      basis=self.basis,
                      var=self.var,
                      regulariser=self.regulariser,
                      tol=self.tol,
                      maxit=self.maxit,
                      verbose=self.verbose
                      )

        return self

    def predict(self, X):

        return self._predict(X)

    def predict_proba(self, X):

        return self._predict(X, uncertainty=True)

    def _predict(self, X, uncertainty=False):

        Ey, Vf, Vy = slm.predict(X,
                                 basis=self.basis,
                                 m=self.m,
                                 C=self.C,
                                 hypers=self.hypers,
                                 var=self.optvar
                                 )

        return (Ey, Vf, Vy) if uncertainty else Ey


class GeneralisedLinearModel(BaseEstimator, RegressorMixin):

    def __init__(self, likelihood, basis, 
                 regulariser=Parameter(1., Positive()), postcomp=10,
                 use_sgd=True, maxit=1000, tol=1e-7, batchsize=100, rho=0.9,
                 epsilon=1e-5, alpha=0.95, verbose=True):

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
        self.alpha = alpha
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

    def predict(self, X):

        Ey, _, _, _ = glm.predict_moments(X,
                                          likelihood=self.likelihood, 
                                          basis=self.basis,
                                          m=self.m,
                                          C=self.C,
                                          lparams=self.lparams,
                                          bparams=self.bparams
                                          )
        return Ey

    def predict_proba(self, X):

        Ey = self.predict(X)
        ql, qu =  glm.predict_interval(alpha=self.alpha,
                                       Xs=X,
                                       likelihood=self.likelihood, 
                                       basis=self.basis,
                                       m=self.m,
                                       C=self.C,
                                       lparams=self.lparams,
                                       bparams=self.bparams
                                       )

        return Ey, ql, qu


class _BaseBasis(BaseEstimator, TransformerMixin, bf.Basis):

    def __init__(self):

        self.kwparams = {}

    def fit(self, X, y=None):
        """ 
        A do-nothing fit function that just maintains compatibility with
        scikit-learn's tranformer interface.
        """

        return self

    def transform(self, X, y=None):

        return self(X, **self.kwparams)

    def fit_transform(self, X, y=None, **fit_params):

        if not fit_params:
            return self.transform(X, y)
        else:
            return self(X, **fit_params)


class LinearBasis(bf.LinearBasis, _BaseBasis):

    def __init__(self, onescol=False):

        self.onescol = onescol
        self.kwparams = {}

        super(LinearBasis, self).__init__(onescol=onescol)


class PolynomialBasis(bf.PolynomialBasis, _BaseBasis):

    def __init__(self, order, include_bias=True):

        self.order = order
        self.include_bias = include_bias
        self.kwparams = {}

        super(PolynomialBasis, self).__init__(order=order,
                                              include_bias=include_bias)


class RandomRBF(bf.RandomRBF, _BaseBasis):

    def __init__(self, nbases, Xdim, lenscale=1):

        self.kwparams = {'lenscale': lenscale}
        self.nbases = nbases
        self.Xdim = Xdim
        self.lenscale = lenscale
        super(RandomRBF, self).__init__(Xdim=Xdim, nbases=nbases)


class RandomLaplace(bf.RandomLaplace, RandomRBF):

    pass


class RandomCauchy(bf.RandomCauchy, RandomRBF):

    pass


class RandomMatern32(bf.RandomMatern32, RandomRBF):

    pass


class RandomMatern52(bf.RandomMatern52, RandomRBF):

    pass
