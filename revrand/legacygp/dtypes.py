from collections import namedtuple

Range = namedtuple('Range', 'lowerBound upperBound initialVal')

QueryParams = namedtuple('GPQuery', 'regressor Xs K_xxs')

RegressionParams = namedtuple('RegressionParams',
                              'X factorisation alpha kernel y')

# Folds = namedtuple('Folds', 'X y flat_y n_folds')
