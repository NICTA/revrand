#!/bin/bash

QUORUM=1
CLUSTERNAME="revrand-spark"
 
docker run -d \
    -e MESOS_QUORUM=${QUORUM} \
    -e MESOS_WORK_DIR=/var/lib/mesos \
    -e MESOS_LOG_DIR=/var/log \
    -e MESOS_CLUSTER=${CLUSTERNAME} \
    --net=host \
    dtpc/mesos mesos-master --no-hostname_lookup
