#! /usr/bin/env python3
""" A La Carte GP Classification example on USPS digits dataset. """

from revrand.utils.datasets import fetch_gpml_usps_resampled_data
from revrand.validation import loglosscat, errrate
from revrand import classification, basis_functions
from sklearn.linear_model import LogisticRegression

import logging

#
# Settings
#

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dig1 = 3
dig2 = 5

# A la Carte classifier setting
nbases = 2000
lenscale = 5
reg = 1e3
doSGD = False
method = 'MAP'
passes = 5000

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
    weights, labels = classification.logistic_sgd(X, Y, Phi, (lenscale,),
                                          regulariser=reg, passes = passes)
elif method == 'MAP':
    weights, labels = classification.logistic_map(X, Y, Phi, (lenscale,),
                                          regulariser=reg)
else:
    raise ValueError("Invalid method chosen!")

lreg = LogisticRegression(penalty='l2')
lreg.fit(X, Y)


# Predict
if method != 'SVI':
    pys_l = classification.logistic_predict(Xs, weights, Phi, (lenscale,))
else:
    pys_l = classification.logistic_mpredict(Xs, weights, C, Phi, bparams)


print("Logistic A La Carte: av nll = {:.6f}, error rate = {:.6f}"
      .format(loglosscat(Ys, pys_l), errrate(Ys, pys_l)))

pys_r = lreg.predict_proba(Xs)
print("Logistic Scikit: av nll = {:.6f}, error rate = {:.6f}"
      .format(loglosscat(Ys, pys_r), errrate(Ys, pys_r)))
