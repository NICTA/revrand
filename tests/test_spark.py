from __future__ import division
import numpy as np
import pytest
from revrand.optimize import sgd, sgd_spark, minimize

hasSparkContext = False
try:
    from pyspark import SparkConf, SparkContext
    conf = (SparkConf()
         .setMaster("local[4]")
         .setAppName("Spark SGD Test")
         .set("spark.executor.memory", "1g"))
    sc = SparkContext(conf = conf)
    sc.addPyFile(__file__)
    hasSparkContext = True
except ImportError:
    pass

@pytest.mark.skipif(not hasSparkContext, reason="Requires a Spark context.")
def test_bounded(make_quadratic):

    a, b, c, data, bounds = make_quadratic
    w0 = np.concatenate((np.random.randn(2), [1.5]))

    res = minimize(qobj, w0, args=(data,), jac=True, bounds=bounds,
                   method='L-BFGS-B')
    Ea_bfgs, Eb_bfgs, Ec_bfgs = res['x']

    res = sgd(qobj, w0, data, bounds=bounds, eval_obj=True, gtol=1e-4,
              passes=1000, rate=0.95, eta=1e-6)
    Ea_sgd, Eb_sgd, Ec_sgd = res['x']

    rdd_partitions = 4
    rdd = sc.parallelize(data,rdd_partitions)
    res = sgd_spark(qobj, w0, rdd, bounds=bounds, batchsize=1000, eval_obj=True, gtol=1e-4,
              passes=1000, rate=0.95, eta=1e-6)
    Ea_sgds, Eb_sgds, Ec_sgds = res['x']

    assert np.allclose((Ea_sgd,  Eb_sgd,  Ec_sgd),
                       (Ea_sgds, Eb_sgds, Ec_sgds),
                       atol=1e-2, rtol=0)

    assert np.allclose((Ea_bfgs, Eb_bfgs, Ec_bfgs),
                       (Ea_sgds, Eb_sgds, Ec_sgds),
                       atol=1e-2, rtol=0)


def qobj(w, data, grad=True):

    y, x = data[:, 0], data[:, 1]
    N = len(data)
    a, b, c = w

    u = y - (a * x**2 + b * x + c)
    f = (u**2).sum() / N
    df = -2 * np.array([(x**2 * u).sum(), (x * u).sum(), u.sum()]) / N

    if grad:
        return f, df
    else:
        return f


if __name__ == "__main__":
    from conftest import make_quadratic
    test_bounded(make_quadratic())
