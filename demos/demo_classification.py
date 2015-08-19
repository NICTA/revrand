#! /usr/bin/env python3
""" A La Carte GP Classification example on USPS digits dataset. """

import os
import wget
import logging
import numpy as np
from subprocess import call
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from pyalacarte import classification, bases
from pyalacarte.validation import logloss, errrate


#
# Settings
#

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dig1 = 3
dig2 = 5

# A la Carte classifier setting
nbases = 1000
lenscale = 5
reg = 1e3


#
# Load data
#

# Pull this data down and process if not present
dpath = 'usps_resampled/usps_resampled.mat'
if not os.path.exists(dpath):

    wget.download('http://www.gaussianprocess.org/gpml/data/usps_resampled'
                  '.tar.bz2')
    call(['tar', '-xjf', 'usps_resampled.tar.bz2'])

# Extract data
data = loadmat('usps_resampled/usps_resampled.mat')
ind1 = data['train_labels'][dig1, :] == 1
ind2 = data['train_labels'][dig2, :] == 1
X = np.hstack((data['train_patterns'][:, ind1],
               data['train_patterns'][:, ind2])).T
Y = np.concatenate((np.ones(ind1.sum()), np.zeros(ind2.sum())))

ind1 = data['test_labels'][dig1, :] == 1
ind2 = data['test_labels'][dig2, :] == 1
Xs = np.hstack((data['test_patterns'][:, ind1],
                data['test_patterns'][:, ind2])).T
Ys = np.concatenate((np.ones(ind1.sum()), np.zeros(ind2.sum())))

# Classify

# Phi = bases.LinearBasis(True)  # + bases.RandomRBF(nbases, X.shape[1])
Phi = bases.RandomRBF(nbases, X.shape[1])
# weights = classification.logistic_map(X, Y, Phi, (lenscale,), regulariser=reg)
# weights = classification.logistic_map(X, Y, Phi, (), regulariser=reg)
# weights = classification.logistic_sgd(X, Y, Phi, (), regulariser=reg)
weights = classification.logistic_sgd(X, Y, Phi, (lenscale,), regulariser=reg)

lreg = LogisticRegression(penalty='l2')
lreg.fit(X, Y)

# Predict
pys_l = classification.logistic_predict(Xs, weights, Phi, (lenscale,))
# pys_l = classification.logistic_predict(Xs, weights, Phi, ())
print("Logistic A La Carte: av nll = {:.6f}, error rate = {:.6f}"
      .format(logloss(Ys, pys_l), errrate(Ys, pys_l)))

pys_r = lreg.predict_proba(Xs)[:, 1]
print("Logistic Scikit: av nll = {:.6f}, error rate = {:.6f}"
      .format(logloss(Ys, pys_r), errrate(Ys, pys_r)))
