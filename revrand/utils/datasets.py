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
from unipath import Path
from io import BytesIO

from .base import Bunch


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
    TODO: Make a little bit more DRY...

    >>> gpml_sarcos = fetch_gpml_sarcos_data()

    >>> gpml_sarcos.train.data.shape
    (44484, 27)

    >>> gpml_sarcos.train.targets.shape
    (44484,)

    >>> gpml_sarcos.train.targets.round(2) # doctest: +ELLIPSIS
    array([ 8.09,  7.76,  7.29, ...,  1.89,  2.49,  2.86])

    >>> gpml_sarcos.test.data.shape
    (4449, 27)

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

    train_bunch = Bunch(data=train_data[:, :-1],
                        targets=train_data[:, -1])

    test_bunch = Bunch(data=test_data[:, :-1],
                       targets=test_data[:, -1])

    return Bunch(train=train_bunch, test=test_bunch)


def fetch_gpml_usps_resampled_data(transpose_data=True, data_home=None):
    """
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
