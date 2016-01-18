#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: run-slave.sh <slave_ip> <master_ip>"
    exit 1
fi

HOSTNAME=$1
MASTER_IP=$2
EXECUTER_REGISTRATION_TIMEOUT="5mins"
CONTAINERIZERS="docker,mesos"

# Configure mesos via files in /etc/mesos-slave
echo ${HOSTNAME} | sudo tee /etc/mesos-slave/hostname
echo "${MASTER_IP}:5050" | sudo tee /etc/mesos-slave/master 
echo ${EXECUTER_REGISTRATION_TIMEOUT} | sudo tee /etc/mesos-slave/executor_registration_timeout
echo ${CONTAINERIZERS} | sudo tee /etc/mesos-slave/containerizers
sudo touch /etc/mesos-slave/?no-hostname_lookup
echo "" | sudo tee /etc/mesos/zk


export SPARK_PUBLIC_DNS=`wget -q -O - http://instance-data.ec2.internal/latest/meta-data/public-hostname`
export MESOS_PUBLIC_DNS=`wget -q -O - http://instance-data.ec2.internal/latest/meta-data/public-hostname`

# Start mesos-slave
sudo service mesos-slave start 

