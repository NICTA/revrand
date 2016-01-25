#!/bin/bash

if [ "$#" -ne 1 ]; then
   echo "Specify cluster accessible ip address of Mesos master"
   exit 1
fi

MESOS_MASTER=$1
PYSPARK_PYTHON=python3 PYSPARK_DRIVER_PYTHON=ipython3 ${SPARK_HOME}/bin/pyspark --master mesos://${MESOS_MASTER}:5050
