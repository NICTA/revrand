#!/bin/bash

MESOS_VERSION="0.26.0-0.2.145.ubuntu1404"
 
sudo docker run -d \
    -e MESOS_LOG_DIR=/var/log/mesos \
    -e MESOS_MASTER=${MESOS_MASTER} \
    -e MESOS_EXECUTOR_REGISTRATION_TIMEOUT=5mins \
    -e MESOS_ISOLATOR=cgroups/cpu,cgroups/mem \
    -e MESOS_CONTAINERIZERS=docker,mesos \
    -e MESOS_PORT=5051 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -v /sys:/sys:ro \
    -v /usr/lib/x86_64-linux-gnu/libapparmor.so.1.1.0:/lib/x86_64-linux-gnu/libapparmor.so.1 \
    --net=host \
    mesosphere/mesos-slave:${MESOS_VERSION} --no-hostname_lookup
  
