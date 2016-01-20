FROM alistaireid/statbuntu:latest

MAINTAINER david.cole@nicta.com.au

RUN apt-get update -y

##### Mesos #####
RUN apt-get install -y lsb-release

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv E56151BF \
 && DISTRO=$(lsb_release -is | tr '[:upper:]' '[:lower:]') \
 && CODENAME=$(lsb_release -cs) \
 && echo "deb http://repos.mesosphere.io/${DISTRO} ${CODENAME} main" | \
    tee /etc/apt/sources.list.d/mesosphere.list \
 && apt-get update -y \
 && apt-get install -y mesos

ENV MESOS_NATIVE_JAVA_LIBRARY /usr/local/lib/libmesos.so
ENV MESOS_WORK_DIR=/var/lib/mesos
ENV MESOS_LOG_DIR=/var/log/mesos
ENV MESOS_LOGGING_LEVEL=INFO

##### Apache Spark
ENV SPARK_VERSION 1.6.0-bin-hadoop2.6
ENV SPARK_HOME /opt/spark
ENV PYSPARK_PYTHON=python3

RUN apt-get install -y \
    curl \
    openjdk-8-jre-headless

RUN curl -sf "http://d3kbcqa49mib13.cloudfront.net/spark-${SPARK_VERSION}.tgz" | tar zx -C /opt && \
    mv "/opt/spark-${SPARK_VERSION}" ${SPARK_HOME}

ADD files/spark-defaults.conf ${SPARK_HOME}/conf/spark-defaults.conf

WORKDIR ${SPARK_HOME}

##### Revrand (spark_sgd branch)
ENV REVRAND_GIT_REF=spark_sgd
RUN pip3 install https://github.com/NICTA/revrand/zipball/${REVRAND_GIT_REF} \
 && pip3 install unipath

##### Clean up
RUN apt-get -y purge lsb-release \
 && apt-get -y autoremove \
 && apt-get -y clean

