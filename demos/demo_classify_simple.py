#! /usr/bin/env python3
""" A La Carte GP Classification example on a simple square wave. """

import logging
import numpy as np
import matplotlib.pyplot as pl
from revrand import classification, basis_functions


#
# Settings
#

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# A la Carte classifier setting
nbases = 500
lenscale = 0.2
reg = 1000
# method = 'SGD'
method = 'MAP'
batchsize = 100
rate = 0.9
eta = 1e-6
passes = 1000

# Dataset Settings
Ntrain = 200
Npred = 3000


# Gen Data
X = np.linspace(-2 * np.pi, 2 * np.pi, Ntrain)[:, np.newaxis]
f = np.sin(X).flatten()
Y = np.round((f + 1) / 2)
Xs = np.linspace(-2.5 * np.pi, 2.5 * np.pi, Npred)[:, np.newaxis]


# Train
Phi = basis_functions.RandomRBF(nbases, X.shape[1])
if method == 'SGD':
    weights, l = classification.learn_sgd(X, Y, Phi, (lenscale,),
                                          regulariser=reg, eta=eta,
                                          batchsize=batchsize, rate=rate,
                                          passes=passes)
elif method == 'MAP':
    weights, l = classification.learn_map(X, Y, Phi, (lenscale,),
                                          regulariser=reg)
else:
    raise ValueError("Invalid method chosen!")


# Predict
Ey = classification.predict(Xs, weights, Phi, (lenscale,))


# Plot
pl.figure()
ax = pl.subplot(111)
pl.plot(Xs, Ey[:, 1], label='Prediction')
pl.plot(X, Y, 'k.', linewidth=2, label='Training data')
pl.grid(True)
pl.title('Simple Square Wave Classification Test')
pl.xlabel('X')
pl.ylabel('p(y* = 1)')
ax.set_ylim(-0.05, 1.05)
pl.legend()
pl.show()
