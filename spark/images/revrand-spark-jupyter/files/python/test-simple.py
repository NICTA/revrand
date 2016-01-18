#! /usr/bin/env python3

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# if demo run within spark environment, create context
hasSparkContext = False
try:
    from pyspark import SparkConf, SparkContext
    conf = (SparkConf()
         .setAppName("Spark regression Demo")
         .set("spark.executor.memory", "1g"))
    sc = SparkContext(conf = conf)
    sc.addPyFile(__file__)
    hasSparkContext = True
except ImportError:
    pass


def main():

    if not hasSparkContext:
        log.error("No spark context")

    else:
        data = range(1000)
        n_partitions = 4
        rdd = sc.parallelize(data, n_partitions)
        x = rdd.map(lambda x: x*2).sum()
        log.info("x = {}".format(x))

if __name__ == "__main__":
    log.info("Simple spark test")
    main()
