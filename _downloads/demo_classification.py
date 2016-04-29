#! /usr/bin/env python3
""" Bayesian GLM Classification example on USPS digits dataset. """

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from revrand.utils.datasets import fetch_gpml_usps_resampled_data
from revrand import glm
from revrand.btypes import Parameter, Positive
from revrand.basis_functions import RandomRBF
from revrand.likelihoods import Bernoulli

import logging

#
# Settings
#

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dig1 = 3
dig2 = 5

# A la Carte classifier setting
nbases = 200
lenscale = 10
doSGD = True
passes = 30
batchsize = 10

#
# Load data
#

usps_resampled = fetch_gpml_usps_resampled_data()

# Training dataset
ind1 = usps_resampled.train.targets == dig1
ind2 = usps_resampled.train.targets == dig2

X = usps_resampled.train.data[np.logical_or(ind1, ind2)]

usps_resampled.train.targets[ind1] = 1
usps_resampled.train.targets[ind2] = 0

Y = usps_resampled.train.targets[np.logical_or(ind1, ind2)]

# Test dataset
ind1 = usps_resampled.test.targets == dig1
ind2 = usps_resampled.test.targets == dig2

Xs = usps_resampled.test.data[np.logical_or(ind1, ind2)]

usps_resampled.test.targets[ind1] = 1
usps_resampled.test.targets[ind2] = 0

Ys = usps_resampled.test.targets[np.logical_or(ind1, ind2)]

# Classify - Revrand
Phi = RandomRBF(nbases, X.shape[1],
                lenscale_init=Parameter(lenscale, Positive()))
llhood = Bernoulli()
params = glm.learn(X, Y, llhood, Phi, use_sgd=doSGD,
                   maxit=passes, batchsize=batchsize)

# Predict
pys_l, Vpy, Epn, Epx = glm.predict_moments(Xs, llhood, Phi, *params)
pys_l = np.vstack((1 - pys_l, pys_l)).T
Eys_l = pys_l[:, 0] >= 0.5

# Classify - Sklearn
lreg = LogisticRegression(penalty='l2', class_weight='balanced')
lreg.fit(Phi(X, lenscale), Y)
pys_r = lreg.predict_proba(Phi(Xs, lenscale))
Eys_r = pys_r[:, 0] >= 0.5

print("GLM: av nll = {:.6f}, error rate = {:.6f}"
      .format(log_loss(Ys, pys_l), accuracy_score(Ys, Eys_l)))

print("Logistic Scikit: av nll = {:.6f}, error rate = {:.6f}"
      .format(log_loss(Ys, pys_r), accuracy_score(Ys, Eys_r)))
