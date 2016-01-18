#! /usr/bin/env python3
""" A La Carte GP Classification example on USPS digits dataset. """

import numpy as np
from sklearn.linear_model import LogisticRegression

from revrand.utils.datasets import fetch_gpml_usps_resampled_data
from revrand.validation import loglosscat, errrate
from revrand import classification, basis_functions, glm, likelihoods

import logging

#
# Settings
#

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dig1 = 3
dig2 = 5

# A la Carte classifier setting
nbases = 300
lenscale = 5
reg = 1
doSGD = True
method = 'GLM'
passes = 300
K = 5

#
# Load data
#

usps_resampled = fetch_gpml_usps_resampled_data()

# Training dataset
ind1 = usps_resampled.train.targets == dig1
ind2 = usps_resampled.train.targets == dig2

X = usps_resampled.train.data[ind1 | ind2]

usps_resampled.train.targets[ind1] = 1
usps_resampled.train.targets[ind2] = 0

Y = usps_resampled.train.targets[ind1 | ind2]

# Test dataset
ind1 = usps_resampled.test.targets == dig1
ind2 = usps_resampled.test.targets == dig2

Xs = usps_resampled.test.data[ind1 | ind2]

usps_resampled.test.targets[ind1] = 1
usps_resampled.test.targets[ind2] = 0

Ys = usps_resampled.test.targets[ind1 | ind2]

# Classify
Phi = basis_functions.RandomRBF(nbases, X.shape[1])
if method == 'SGD':
    weights, labels = classification.learn_sgd(X, Y, Phi, (lenscale,),
                                               regulariser=reg,
                                               passes=passes)
elif method == 'MAP':
    weights, labels = classification.learn_map(X, Y, Phi, (lenscale,),
                                               regulariser=reg)
elif method == 'GLM':
    llhood = likelihoods.Bernoulli()
    lparams = []
    params = glm.learn(X, Y, llhood, lparams, Phi, [lenscale], reg=reg,
                       use_sgd=doSGD, maxit=passes, postcomp=K)
else:
    raise ValueError("Invalid method chosen!")

lreg = LogisticRegression(penalty='l2', class_weight='balanced', C=reg)
lreg.fit(Phi(X, lenscale), Y)


# Predict
if method == 'GLM':
    pys_l, Vpy, Epn, Epx = glm.predict_meanvar(Xs, llhood, Phi, *params)
    pys_l = np.vstack((1 - pys_l, pys_l)).T
else:
    pys_l = classification.predict(Xs, weights, Phi, (lenscale,))

print("Logistic {}: av nll = {:.6f}, error rate = {:.6f}"
      .format(method, loglosscat(Ys, pys_l), errrate(Ys, pys_l)))

pys_r = lreg.predict_proba(Phi(Xs, lenscale))
print("Logistic Scikit: av nll = {:.6f}, error rate = {:.6f}"
      .format(loglosscat(Ys, pys_r), errrate(Ys, pys_r)))
