#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Specify cluster accessible address <host:port> of Mesos master"
    exit 1
fi

MESOS_MASTER=$1
sudo docker run -d \
    -e PYSPARK_SUBMIT_ARGS="--master=mesos://${MESOS_MASTER}:5050 pyspark-shell" \
    --net=host dtpc/revrand-spark-jupyter \
    jupyter notebook

