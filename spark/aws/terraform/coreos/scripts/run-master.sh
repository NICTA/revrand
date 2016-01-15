#!/bin/bash

QUORUM=1
CLUSTERNAME="revrand-spark"
MESOS_VERSION="0.26.0-0.2.145.ubuntu1404"

 
docker run -d \
    -e MESOS_QUORUM=${QUORUM} \
    -e MESOS_WORK_DIR=/var/lib/mesos \
    -e MESOS_LOG_DIR=/var/log \
    -e MESOS_CLUSTER=${CLUSTERNAME} \
    --net=host \
    mesosphere/mesos-master:${MESOS_VERSION} --no-hostname_lookup
