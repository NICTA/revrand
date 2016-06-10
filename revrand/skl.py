"""
Scikit learn interface - for making revrand compatible with pipelines.

Note
----
You cannot use the basis function objects in this module with the regressors if
you wish to learn their parameters, they are simply included as transformation
objects. You can still use the original basis function objects with the
regression wrappers here though.

"""

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from . import slm, glm
from . import basis_functions as bf
from .btypes import Parameter, Positive


class StandardLinearModel(BaseEstimator, RegressorMixin):
    """
    Standard linear model interface class.

    This provides a scikit learn compatible interface for the slm module.

    Parameters
    ----------
        basis: Basis
            A basis object, see the basis_functions module.
        var: Parameter, optional
            observation variance initial value.
        regulariser: Parameter, optional
            weight regulariser (variance) initial value.
        tol: float, optional
            optimiser function tolerance convergence criterion.
        maxit: int, optional
            maximum number of iterations for the optimiser.
    """

    def __init__(self, basis, var=Parameter(1., Positive()),
                 regulariser=Parameter(1., Positive()), tol=1e-6, maxit=500):

        self.basis = basis
        self.var = var
        self.regulariser = regulariser
        self.tol = tol
        self.maxit = maxit

    def fit(self, X, y):
        """
        Learn the parameters and hyperparameters of a Bayesian linear
        regressor.

        Parameters
        ----------
            X: ndarray
                (N, d) array input dataset (N samples, d dimensions).
            y: ndarray
                (N,) array targets (N samples)
        """

        self.m, self.C, self.hypers, self.optvar = \
            slm.learn(X, y,
                      basis=self.basis,
                      var=self.var,
                      regulariser=self.regulariser,
                      tol=self.tol,
                      maxit=self.maxit,
                      )

        return self

    def predict(self, X):
        """
        Predict mean from Bayesian linear regression.

        Parameters
        ----------
            X: ndarray
                (Ns,d) array query input dataset (Ns samples, d dimensions).

        Returns
        -------
            Ey: ndarray
                The expected value of y_star for the query inputs, X_star
                of shape (N_star,).
        """

        return self._predict(X)

    def predict_proba(self, X):
        """
        Full predictive distribution from Bayesian linear regression.

        Parameters
        ----------
            X: ndarray
                (Ns,d) array query input dataset (Ns samples, d dimensions).

        Returns
        -------
            Ey: ndarray
                The expected value of y_star for the query inputs, X_star
                of shape (N_star,).
            Vf: ndarray
                The expected variance of f_star for the query inputs,
                X_star of shape (N_star,).
            Vy: ndarray
                The expected variance of y_star for the query inputs,
                X_star of shape (N_star,).
        """

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
    """
    Generalised linear model interface class.

    This provides a scikit learn compatible interface for the glm module.

    Parameters
    ----------
        likelihood: Object
            A likelihood object, see the likelihoods module.
        basis: Basis
            A basis object, see the basis_functions module.
        regulariser: Parameter, optional
            weight regulariser (variance) initial value.
        postcomp: int, optional
            Number of diagonal Gaussian components to use to approximate the
            posterior distribution.
        tol: float, optional
           Optimiser relative tolerance convergence criterion.
        use_sgd: bool, optional
            If :code:`True` then use SGD (Adadelta) optimisation instead of
            L-BFGS.
        maxit: int, optional
            Maximum number of iterations of the optimiser to run. If
            :code:`use_sgd` is :code:`True` then this is the number of complete
            passes through the data before optimization terminates (unless it
            converges first).
        batchsize: int, optional
            number of observations to use per SGD batch. Ignored if
            :code:`use_sgd=False`.
        rho: float, optional
            SGD decay rate, must be [0, 1]. Ignored if :code:`use_sgd=False`.
        epsilon: float, optional
            Jitter term for adadelta SGD. Ignored if :code:`use_sgd=False`.
        alpha: float, optional
            The percentile confidence interval (e.g. 95%) to return from
            predict_proba.
    """

    def __init__(self, likelihood, basis,
                 regulariser=Parameter(1., Positive()), postcomp=10,
                 use_sgd=True, maxit=1000, tol=1e-7, batchsize=100, rho=0.9,
                 epsilon=1e-5, alpha=0.95):

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

    def fit(self, X, y):
        """

        Parameters
        ----------
            X: ndarray
                (N, d) array input dataset (N samples, d dimensions).
            y: ndarray
                (N,) array targets (N samples)
        """

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
                      epsilon=self.epsilon)

        return self

    def predict(self, X):
        """
        Predict the target expected value from the generalised linear model.

        Parameters
        ----------
            X: ndarray
                (Ns,d) array query input dataset (Ns samples, d dimensions).

        Returns
        -------
            Ey: ndarray
                The expected value of y_star for the query inputs, X_star
                of shape (N_star,).
        """

        Ey, _, _, _ = glm.predict_moments(X,
                                          likelihood=self.likelihood,
                                          basis=self.basis,
                                          m=self.m,
                                          C=self.C,
                                          lparams=self.lparams,
                                          bparams=self.bparams
                                          )
        return Ey

    def predict_proba(self, X, alpha=None):
        """
        Predicted target value and uncertainty quantiles from the generalised
        linear model.

        Parameters
        ----------
            X: ndarray
                (Ns,d) array query input dataset (Ns samples, d dimensions).
            alpha: float, optional
                The percentile confidence interval (e.g. 95%) to return from
                predict_proba. If this is None, the value in the constructor is
                used.

        Returns
        -------
            Ey: ndarray
                The expected value of y_star for the query inputs, X_star
                of shape (N_star,).
            ql: ndarray
                The lower end point of the interval with shape (N_star,)
            qu: ndarray
                The upper end point of the interval with shape (N_star,)
        """

        if alpha is not None:
            self.alpha = alpha

        Ey = self.predict(X)
        ql, qu = glm.predict_interval(alpha=self.alpha,
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
        """
        Apply the basis function to X.

        Parameters
        ----------
        X: ndarray
            of shape (N, d) of observations where N is the number of samples,
            and d is the dimensionality of X.

        Returns
        -------
        ndarray:
            The basis function transform (Phi).
        """

        return self(X, **self.kwparams)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Apply the basis function to X.

        Parameters
        ----------
        X: ndarray
            of shape (N, d) of observations where N is the number of samples,
            and d is the dimensionality of X.
        fit_params: sequence
            extra parameters to supply the basis function (see
            basis_functions.py).

        Returns
        -------
        ndarray:
            The basis function transform (Phi).

        Note
        ----
        This is the same as :code:`transform()` with the option of passing in
        extra parameters.
        """

        if not fit_params:
            return self.transform(X, y)
        else:
            return self(X, **fit_params)


class LinearBasis(bf.LinearBasis, _BaseBasis):
    """
    Linear basis class, basically this just prepends a columns of ones onto X

    Parameters
    ----------
    onescol: bool, optional
        If true, prepend a column of ones onto X.
    """

    def __init__(self, onescol=False):

        self.onescol = onescol
        self.kwparams = {}

        super(LinearBasis, self).__init__(onescol=onescol)


class PolynomialBasis(bf.PolynomialBasis, _BaseBasis):
    """
    Polynomial basis class, this essentially creates the concatenation,
    :math:`\\boldsymbol\Phi = [\mathbf{X}^0, \mathbf{X}^1, ..., \mathbf{X}^p]`
    where :math:`p` is the :code:`order` of the polynomial.

    Parameters
    ----------
    order: int
        the order of the polynomial to create, i.e. the last power to raise
        X to in the concatenation Phi = [X^0, X^1, ..., X^order].
    include_bias: bool, optional
        If True (default), include the bias column (column of ones which
        acts as the intercept term in a linear model)
    """

    def __init__(self, order, include_bias=True):

        self.order = order
        self.include_bias = include_bias
        self.kwparams = {}

        super(PolynomialBasis, self).__init__(order=order,
                                              include_bias=include_bias)


class RadialBasis(bf.RadialBasis, _BaseBasis):
    """
    Radial basis class.

    Parameters
    ----------
    centres: ndarray
        array of shape (Dxd) where D is the number of centres for the
        radial bases, and d is the dimensionality of X.
    lenscale: float
        the length scale (scalar) of the RBFs to apply to X.

    Note
    ----
    This will have relevance vector machine-like behaviour with uncertainty.
    """

    def __init__(self, centres, lenscale):

        self.centres = centres
        self.lenscale = lenscale
        self.kwparams = {'lenscale': lenscale}

        super(RadialBasis, self).__init__(centres=centres)


class SigmoidalBasis(bf.SigmoidalBasis, RadialBasis):
    """
    Sigmoidal Basis

    Parameters
    ----------
    centres: ndarray
        array of shape (Dxd) where D is the number of centres for the bases,
        and d is the dimensionality of X.
    lenscale: float
        the length scale (scalar) of the RBFs to apply to X.
    """

    pass


class RandomRBF(bf.RandomRBF, _BaseBasis):
    """
    Random RBF Basis -- Approximates an RBF kernel function

    This will make a linear regression model approximate a GP with an
    (optionally ARD) RBF covariance function.

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations
    lenscale: float
        the length scale (scalar) of the RBFs to apply to X.
    """

    def __init__(self, nbases, Xdim, lenscale):

        self.nbases = nbases
        self.Xdim = Xdim
        self.lenscale = lenscale
        self.kwparams = {'lenscale': lenscale}
        super(RandomRBF, self).__init__(Xdim=Xdim, nbases=nbases)


class RandomLaplace(bf.RandomLaplace, RandomRBF):
    """
    Random Laplace Basis -- Approximates a Laplace kernel function

    This will make a linear regression model approximate a GP with an
    (optionally ARD) Laplace covariance function.

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations
    lenscale: float
        the length scale (scalar) of the RBFs to apply to X.
    """
    pass


class RandomCauchy(bf.RandomCauchy, RandomRBF):
    """
    Random Cauchy Basis -- Approximates a Cauchy kernel function

    This will make a linear regression model approximate a GP with an
    (optionally ARD) Laplace covariance function.

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations
    lenscale: float
        the length scale (scalar) of the RBFs to apply to X.
    """

    pass


class RandomMatern32(bf.RandomMatern32, RandomRBF):
    """
    Random Matern 3/2 Basis -- Approximates a Matern 3/2 kernel function

    This will make a linear regression model approximate a GP with an
    (optionally ARD) Laplace covariance function.

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations
    lenscale: float
        the length scale (scalar) of the RBFs to apply to X.
    """
    pass


class RandomMatern52(bf.RandomMatern52, RandomRBF):
    """
    Random Matern 5/2 Basis -- Approximates a Matern 5/2 kernel function

    This will make a linear regression model approximate a GP with an
    (optionally ARD) Laplace covariance function.

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations
    lenscale: float
        the length scale (scalar) of the RBFs to apply to X.
    """

    pass
