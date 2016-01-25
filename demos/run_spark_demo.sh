#!/bin/bash

# Run a demo script with a PySpark backend 
# Demos which can utilise spark to run distributed algorithms: 
#   - demo_sgd_spark.py
#   - demo_glm.py
#   - demo_regression.py

if [ "$#" -lt 1 ]; then
    echo "usage: run_spark_demp.sh <python_script> [<master>]"
    exit 1
fi

DEMO_SCRIPT=$1
LOCAL="local[*]"
MASTER=${2:-$LOCAL}

PYSPARK_PYTHON=python3 ${SPARK_HOME}/bin/spark-submit --master "${MASTER}" --verbose $DEMO_SCRIPT

