#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Specify cluster accessible address <host:port> of Mesos master, and number of slaves"
    exit 1
fi

MESOS_MASTER=$1
N_SLAVES=$2

aws ecs run-task --cluster spark_mesos --task-definition mesos_slave --count 2 --overrides "{ \"containerOverrides\": [ { \"name\": \"mesos_slave\", \"environment\": [ {  \"name\": \"MESOS_MASTER\", \"value\": \"${MESOS_MASTER}\" } ] } ] }"



