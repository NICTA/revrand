FROM dtpc/revrand-spark:latest

MAINTAINER david.cole@nicta.com.au

##### Jupyter notebook
RUN pip3 install jupyter \
 && pip3 install py4j

ADD files/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
ADD files/jupyter_pyspark_kernel.json /usr/local/share/jupyter/kernels/pyspark/kernel.json
ADD files/run-jupyter.sh /root/run-jupyter.sh

##### vim
RUN apt-get -y update \
 && apt-get install -y \
    vim-tiny \
    curl

##### SGD demo requires GP from Dora

### nlopt
RUN curl -sf http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz | tar zx -C /root \
 && cd /root/nlopt-2.4.2 \
 && ./configure PYTHON=python3 --enable-shared LDFLAGS="-Wl,--rpath=/usr/local/lib" \
 && make \
 && make install \
 && cp -r /usr/local/lib/python3.4/site-packages/* /usr/local/lib/python3.4/dist-packages/ \
 && rm -r /usr/local/lib/python3.4/site-packages \
 && rm -r /root/nlopt-2.4.2

### 
ADD files/test-simple.py /root/test-simple.py
ADD files/spark-demo.py /root/spark-demo.py
ADD files/run-demo.sh /root/run-demo.sh

ADD files/dora-0.1-py3.4.egg /root/dora-0.1-py3.4.egg

### 
RUN pip3 install sklearn \
 && pip3 install flask





