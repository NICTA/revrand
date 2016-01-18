#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Specify cluster accessible ip address of Mesos master"
    exit 1
fi

MESOS_MASTER=$1
${SPARK_HOME}/bin/spark-submit  --master mesos://${MESOS_MASTER}:5050 --py-files=/root/dora-0.1-py3.4.egg /root/spark-demo.py
