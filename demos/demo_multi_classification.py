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
from pyalacarte.validation import loglosscat, errrate


#
# Settings
#

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# A la Carte classifier setting
nbases = 10000
lenscale = 5
reg = 1e3
doSGD = False
method = 'MAP'
numdigits = 3

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
X = data['train_patterns'].T
Y = np.asarray([np.argmax(y) for y in data['train_labels'].T])

Xs = data['test_patterns'].T
Ys = np.asarray([np.argmax(y) for y in data['test_labels'].T])

# Sort and Remove excess labels (specified by numdigits)
sorted_idx = np.argsort(Y)
X = X[sorted_idx,:]
Y = Y[sorted_idx]
sorted_idx_s = np.argsort(Ys)
Xs = Xs[sorted_idx_s,:]
Ys = Ys[sorted_idx_s]
end_id= np.argwhere(Y==numdigits)[0][0]
X = X[:end_id,:]
Y = Y[:end_id]
end_id_s= np.where(Ys==numdigits)[0][0]
Xs = Xs[:end_id_s,:]
Ys = Ys[:end_id_s]


# Classify
Phi = bases.RandomRBF(nbases, X.shape[1])
if method == 'SGD':
    weights, labels = classification.logistic_sgd(X, Y, Phi, (lenscale,),
                                          regulariser=reg)
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
