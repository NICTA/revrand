#!/bin/bash

# Run a demo script with a PySpark backend 
# Demos which can utilise spark to run distributed algorithms: 
#   - demo_sgd_spark.py
#   - demo_glm.py
#   - demo_regression.py

if [ "$#" -ne 1 ]; then
    echo "usage: run_spark_demp.sh <python_script>"
    exit 1
fi

DEMO_SCRIPT=$1

# local master using all cores
MASTER="local[*]" 

PYSPARK_SUBMIT_ARGS="--master ${MASTER}" PYSPARK_PYTHON=python3 spark-submit $DEMO_SCRIPT

