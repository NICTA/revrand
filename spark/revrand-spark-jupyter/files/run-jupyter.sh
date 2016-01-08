#!/bin/bash

if [ "$#" -ne 1 ]; then
   echo "Specify cluster accessible ip address of Mesos master"
   exit 1
fi

MESOS_MASTER=$1
PYSPARK_SUBMIT_ARGS="--master=mesos://${MESOS_MASTER}:5050 pyspark-shell" jupyter notebook

