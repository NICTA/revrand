FROM ubuntu:14.04

ENV DEBIAN_FRONTEND noninteractive

## Mesos
RUN echo "deb http://repos.mesosphere.io/ubuntu/ trusty main" > /etc/apt/sources.list.d/mesosphere.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv E56151BF
RUN apt-get -y update
RUN apt-get -y install mesos

## Docker
RUN apt-get -y install docker.io

