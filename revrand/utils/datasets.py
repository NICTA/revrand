"""
Dataset loading utilities

Portions of this module derived from
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
"""

import numpy as np
import requests
import tarfile
import os

from six.moves import urllib
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from unipath import Path
from io import BytesIO
from sklearn.utils import check_random_state

from .base import Bunch


def make_regression(func, n_samples=100, n_features=1, bias=0.0, noise=0.0,
                    random_state=None):
    """
    Make dataset for a regression problem.

    Examples
    --------
    >>> f = lambda x: 0.5*x + np.sin(2*x)
    >>> X, y = make_regression(f, bias=.5, noise=1., random_state=1)
    >>> X.shape
    (100, 1)
    >>> y.shape
    (100,)
    >>> X[:5].round(2)
    array([[ 1.62],
           [-0.61],
           [-0.53],
           [-1.07],
           [ 0.87]])
    >>> y[:5].round(2)
    array([ 0.76,  0.48, -0.23, -0.28,  0.83])
    """
    generator = check_random_state(random_state)

    X = generator.randn(n_samples, n_features)
    # unpack the columns of X
    y = func(*X.T) + bias

    if noise > 0.0:
        y += generator.normal(scale=noise, size=y.shape)

    return X, y


def make_polynomial(degree=3, n_samples=100, bias=0.0, noise=0.0,
                    return_coefs=False, random_state=None):
    """
    Generate a noisy polynomial for a regression problem

    Examples
    --------
    >>> X, y, coefs = make_polynomial(degree=3, n_samples=200, noise=.5,
    ...                               return_coefs=True, random_state=1)
    """
    generator = check_random_state(random_state)

    # TODO: Add arguments to support other priors
    coefs = generator.randn(degree + 1)
    pows = np.arange(degree + 1)
    poly = np.vectorize(lambda x: np.sum(coefs * x ** pows))
    X, y = make_regression(poly, n_samples=n_samples, bias=bias, noise=noise,
                           random_state=random_state)
    if return_coefs:
        return X, y, coefs

    return X, y


def get_data_home(data_home=None):
    """
    Return the path of the revrand data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the data dir is set to a folder named 'revrand_data'
    in the user home folder.

    Alternatively, it can be set by the 'REVRAND_DATA' environment
    variable or programmatically by giving an explicit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    data_home_default = Path(__file__).ancestor(3).child('demos',
                                                         '_revrand_data')

    if data_home is None:
        data_home = os.environ.get('REVRAND_DATA', data_home_default)

    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return data_home


def fetch_gpml_sarcos_data(transpose_data=True, data_home=None):
    """
    Fetch the SARCOS dataset from the internet and parse appropriately into
    python arrays

    >>> gpml_sarcos = fetch_gpml_sarcos_data()

    >>> gpml_sarcos.train.data.shape
    (44484, 21)

    >>> gpml_sarcos.train.targets.shape
    (44484,)

    >>> gpml_sarcos.train.targets.round(2) # doctest: +ELLIPSIS
    array([ 50.29,  44.1 ,  37.35, ...,  22.7 ,  17.13,   6.52])

    >>> gpml_sarcos.test.data.shape
    (4449, 21)

    >>> gpml_sarcos.test.targets.shape
    (4449,)
    """
    train_src_url = "http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat"
    test_src_url = ("http://www.gaussianprocess.org/gpml/data/sarcos_inv_test"
                    ".mat")

    data_home = get_data_home(data_home=data_home)

    train_filename = os.path.join(data_home, 'sarcos_inv.mat')
    test_filename = os.path.join(data_home, 'sarcos_inv_test.mat')

    if not os.path.exists(train_filename):
        urllib.request.urlretrieve(train_src_url, train_filename)

    if not os.path.exists(test_filename):
        urllib.request.urlretrieve(test_src_url, test_filename)

    train_data = loadmat(train_filename).get('sarcos_inv')
    test_data = loadmat(test_filename).get('sarcos_inv_test')

    train_bunch = Bunch(data=train_data[:, :21],
                        targets=train_data[:, 21])

    test_bunch = Bunch(data=test_data[:, :21],
                       targets=test_data[:, 21])

    return Bunch(train=train_bunch, test=test_bunch)


def fetch_gpml_usps_resampled_data(transpose_data=True, data_home=None):
    """
    Fetch the USPS handwritten digits dataset from the internet and parse
    appropriately into python arrays

    >>> usps_resampled = fetch_gpml_usps_resampled_data()

    >>> usps_resampled.train.targets.shape
    (4649,)

    >>> usps_resampled.train.targets # doctest: +ELLIPSIS
    array([6, 0, 1, ..., 9, 2, 7])

    >>> usps_resampled.train.data.shape
    (4649, 256)

    >>> np.all(-1 <= usps_resampled.train.data)
    True

    >>> np.all(usps_resampled.train.data < 1)
    True

    >>> usps_resampled.test.targets.shape
    (4649,)

    >>> usps_resampled.test.data.shape
    (4649, 256)

    >>> usps_resampled = fetch_gpml_usps_resampled_data(transpose_data=False)
    >>> usps_resampled.train.data.shape
    (256, 4649)
    """
    data_home = get_data_home(data_home=data_home)
    data_filename = os.path.join(data_home,
                                 'usps_resampled/usps_resampled.mat')

    if not os.path.exists(data_filename):

        r = requests.get('http://www.gaussianprocess.org/gpml/data/'
                         'usps_resampled.tar.bz2')

        with tarfile.open(fileobj=BytesIO(r.content)) as tar_infile:
            tar_infile.extract('usps_resampled/usps_resampled.mat',
                               path=data_home)

    matlab_dict = loadmat(data_filename)

    train_data = matlab_dict['train_patterns']
    test_data = matlab_dict['test_patterns']

    if transpose_data:
        train_data = train_data.T
        test_data = test_data.T

    train_targets = matlab_dict['train_labels'].T
    train_targets = np.argwhere(train_targets == 1)[:, 1]

    test_targets = matlab_dict['test_labels'].T
    test_targets = np.argwhere(test_targets == 1)[:, 1]

    train_bunch = Bunch(data=train_data,
                        targets=train_targets)

    test_bunch = Bunch(data=test_data,
                       targets=test_targets)

    return Bunch(train=train_bunch, test=test_bunch)


def gen_gausprocess_se(ntrain, ntest, noise=1., lenscale=1., scale=1.,
                       xmin=-10, xmax=10):
    """
    Generate a random (noisy) draw from a Gaussian Process with a RBF kernel.
    """

    # Xtrain = np.linspace(xmin, xmax, ntrain)[:, np.newaxis]
    Xtrain = np.random.rand(ntrain)[:, np.newaxis] * (xmin - xmax) - xmin
    Xtest = np.linspace(xmin, xmax, ntest)[:, np.newaxis]
    Xcat = np.vstack((Xtrain, Xtest))

    K = scale * np.exp(-cdist(Xcat, Xcat, metric='sqeuclidean') /
                       (2 * lenscale**2))
    U, S, V = np.linalg.svd(K)
    L = U.dot(np.diag(np.sqrt(S))).dot(V)
    f = np.random.randn(ntrain + ntest).dot(L)

    ytrain = f[0:ntrain] + np.random.randn(ntrain) * noise
    ftest = f[ntrain:]

    return Xtrain, ytrain, Xtest, ftest
