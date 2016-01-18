#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Specify cluster accessible ip address of Mesos master"
    exit 1
fi

MESOS_MASTER=$1
DIR=/root/python
${SPARK_HOME}/bin/spark-submit  --master mesos://${MESOS_MASTER}:5050 --py-files=${DIR}/dora-0.1-py3.4.egg ${DIR}/spark-demo.py
