#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Specify cluster accessible address <host:port> of Mesos master"
    exit 1
fi

MESOS_MASTER=$1 envsubst < jupyter.json.template > .jupyter.json
dcos marathon app add .jupyter.json

