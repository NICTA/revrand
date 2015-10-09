"""
Dataset loading utilities

Portions of this module derived from 
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
"""

import numpy as np
import requests
import tarfile
import os

from scipy.io import loadmat
from io import BytesIO

from .base import Bunch

def get_data_home(data_home=None):
    """
    Return the path of the pyalacarte data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.
    
    By default the data dir is set to a folder named 'pyalacarte_data'
    in the user home folder.

    Alternatively, it can be set by the 'PYALACARTE_DATA' environment
    variable or programmatically by giving an explicit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = os.environ.get('PYALACARTE_DATA',
                                   os.path.join('~', 'pyalacarte_data'))

    data_home = os.path.expanduser(data_home)
    
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    
    return data_home

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
    data_filename = os.path.join(data_home, 'usps_resampled/usps_resampled.mat')

    if not os.path.exists(data_filename):

        r = requests.get('http://www.gaussianprocess.org/gpml/data/usps_resampled'
                         '.tar.bz2', stream=True)

        with tarfile.open(fileobj=BytesIO(r.content)) as tar_infile:
            tar_infile.extract('usps_resampled/usps_resampled.mat', path=data_home) 

    matlab_dict = loadmat(data_filename)

    train_data = matlab_dict['train_patterns']
    test_data  = matlab_dict['test_patterns']

    if transpose_data:
        train_data = train_data.T
        test_data = test_data.T

    train_targets = matlab_dict['train_labels'].T
    train_targets = np.argwhere(train_targets == 1)[:, 1]

    test_targets  = matlab_dict['test_labels'].T
    test_targets  = np.argwhere(test_targets == 1)[:, 1]
    
    train_bunch = Bunch(data=train_data, 
                        targets=train_targets)

    test_bunch = Bunch(data=test_data, 
                       targets=test_targets)

    return Bunch(train=train_bunch, test=test_bunch)
