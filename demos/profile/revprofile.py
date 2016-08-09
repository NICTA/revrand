"""Some code for profiling revrand."""
import logging
import numpy as np
from sklearn.datasets import load_boston
from sklearn.cross_validation import KFold

from revrand import GeneralisedLinearModel
from revrand.likelihoods import Gaussian
from revrand.basis_functions import LinearBasis, RandomMatern32
from revrand.metrics import smse, msll
from revrand.btypes import Parameter, Positive

# Log output to the terminal attached to this notebook
logging.basicConfig(level=logging.INFO)

# Load the data
boston = load_boston()
X = boston.data
y = boston.target - boston.target.mean()

folds = 5
(tr_ind, ts_ind) = list(KFold(len(y), n_folds=folds, shuffle=True))[0]

# Make Basis and Likelihood
N, D = X.shape
lenscale = 10.
nbases = 300
lenARD = lenscale * np.ones(D)
lenscale_init = Parameter(lenARD, Positive())
base = LinearBasis(onescol=True) + RandomMatern32(Xdim=D, nbases=nbases,
                                                  lenscale_init=lenscale_init)
like = Gaussian()

# Fit and predict the model
glm = GeneralisedLinearModel(like, base, maxiter=4000)
glm.fit(X[tr_ind], y[tr_ind])
Ey, Vy, _, _ = glm.predict_moments(X[ts_ind])

# Score
y_true = y[ts_ind]
print("SMSE = {}, MSLL = {}".format(smse(y_true, Ey),
                                    msll(y_true, Ey, Vy, y[tr_ind])))
