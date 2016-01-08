FROM alistaireid/statbuntu:latest
MAINTAINER david.cole@nicta.com.au

##### Revrand (spark_sgd)
RUN pip3 install https://github.com/NICTA/revrand/archive/spark_sgd.zip \
 && pip3 install unipath

##### Apache Spark
ENV SPARK_VERSION 1.6.0-bin-hadoop2.6

RUN apt-get -y update \
 && apt-get install -y \
    curl \
    openjdk-8-jre-headless
RUN curl -sf "http://d3kbcqa49mib13.cloudfront.net/spark-${SPARK_VERSION}.tgz" | tar zx -C /opt && \
    mv "/opt/spark-${SPARK_VERSION}" /opt/spark

ENV SPARK_HOME /opt/spark
ENV PYSPARK_PYTHON=python3
WORKDIR /opt/spark

##### Mesos
RUN apt-get install -y lsb-release

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv E56151BF \
 && DISTRO=$(lsb_release -is | tr '[:upper:]' '[:lower:]') \
 && CODENAME=$(lsb_release -cs) \
 && echo "deb http://repos.mesosphere.io/${DISTRO} ${CODENAME} main" | \
    tee /etc/apt/sources.list.d/mesosphere.list \
 && apt-get -y update \
 && apt-get install -y mesos

ENV MESOS_WORK_DIR=/var/lib/mesos
ENV MESOS_LOG_DIR=/var/log/mesos
ENV MESOS_LOGGING_LEVEL=INFO
ENV MESOS_NATIVE_JAVA_LIBRARY /usr/local/lib/libmesos.so

##### Spark Config + Scripts

ENV SPARK_MASTER_OPTS="-Dspark.driver.port=7001 -Dspark.fileserver.port=7002 -Dspark.broadcast.port=7003 -Dspark.replClassServer.port=7004 -Dspark.blockManager.port=7005 -Dspark.executor.port=7006 -Dspark.ui.port=4040 -Dspark.broadcast.factory=org.apache.spark.broadcast.HttpBroadcastFactory"

ENV SPARK_WORKER_OPTS="-Dspark.driver.port=7001 -Dspark.fileserver.port=7002 -Dspark.broadcast.port=7003 -Dspark.replClassServer.port=7004 -Dspark.blockManager.port=7005 -Dspark.executor.port=7006 -Dspark.ui.port=4040 -Dspark.broadcast.factory=org.apache.spark.broadcast.HttpBroadcastFactory"

ENV SPARK_MASTER_PORT 7077
ENV SPARK_MASTER_WEBUI_PORT 8080
ENV SPARK_WORKER_PORT 8888
ENV SPARK_WORKER_WEBUI_PORT 8081

EXPOSE 8080 7077 8888 8081 4040 7001 7002 7003 7004 7005 7006

ADD scripts/start-master.sh /start-master.sh
ADD scripts/start-worker.sh /start-worker.sh
ADD scripts/pyspark.sh /pyspark.sh
ADD scripts/remove_alias.sh /remove_alias.sh
ADD scripts/docker-run-spark-env.sh /docker-run-spark-env.sh

ADD files/spark-test.py /spark-test.py
ADD files/test-simple.py /test-simple.py
ADD files/dora-0.1-py3.4.egg /dora-0.1-py3.4.egg

ADD files/spark-defaults.conf ${SPARK_HOME}/conf/spark-defaults.conf



##### Needed by dora (for demo driver, not workers) 
ENV PYFILES="/dora-0.1-py3.4.egg"

##### nlopt
RUN curl -sf http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz | tar zx -C /root \
 && cd /root/nlopt-2.4.2 \
 && ./configure PYTHON=python3 --enable-shared LDFLAGS="-Wl,--rpath=/usr/local/lib" \
 && make \
 && make install \
 && cp -r /usr/local/lib/python3.4/site-packages/* /usr/local/lib/python3.4/dist-packages/ \
 && rm -r /usr/local/lib/python3.4/site-packages \
 && rm -r /root/nlopt-2.4.2

RUN pip3 install sklearn \
 && pip3 install flask

#### vim
RUN apt-get install -y vim-tiny

####### ipython
RUN pip3 install jupyter \
 && pip3 install py4j
ADD files/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
ADD files/jupyter_pyspark_kernel.json /usr/local/share/jupyter/kernels/pyspark/kernel.json




