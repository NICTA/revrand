#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Specify cluster accessible address <host:port> of Mesos master"
    exit 1
fi

SPARK_PUBLIC_DNS=`wget -q -O - http://instance-data.ec2.internal/latest/meta-data/public-hostname`
MESOS_MASTER=$1
docker run -d \
    -e SPARK_PUBLIC_DNS=$SPARK_PUBLIC_DNS \
    -e PYSPARK_SUBMIT_ARGS="--master=mesos://${MESOS_MASTER}:5050 pyspark-shell" \
    --net=host dtpc/revrand-spark-jupyter

