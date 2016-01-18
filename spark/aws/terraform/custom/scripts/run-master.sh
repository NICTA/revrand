#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: run-master.sh <master_ip>"
    exit 1
fi

HOSTNAME=$1
USER=ubuntu
CLUSTERNAME="revrand-spark"
WORKDIR=/var/lib/mesos
QUORUM=1
REGISTRY="in_memory"

# Configure mesos via files in /etc/mesos-master
echo ${HOSTNAME} | sudo tee /etc/mesos-master/hostname
echo ${QUORUM} | sudo tee /etc/mesos-master/quorum
echo ${CLUSTERNAME} | sudo tee /etc/mesos-master/cluster
echo ${WORKDIR} | sudo tee /etc/mesos-master/work_dir
echo ${REGISTRY} | sudo tee /etc/mesos-master/registry
sudo touch /etc/mesos-master/?no-hostname_lookup
echo "" | sudo tee /etc/mesos/zk

# this may not do anything
export MESOS_PUBLIC_DNS=`wget -q -O - http://instance-data.ec2.internal/latest/meta-data/public-hostname`
# start mesos master service
sudo service mesos-master start

