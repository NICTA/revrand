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
RUN pip3 install sklearn \
 && pip3 install flask

### Dora
ENV DORA_GIT_REF=master
RUN pip3 install https://github.com/NICTA/dora/zipball/${DORA_GIT_REF}

# Install Tini
RUN apt-get install -y wget 
RUN wget --quiet https://github.com/krallin/tini/releases/download/v0.6.0/tini && \
    echo "d5ed732199c36a1189320e6c4859f0169e950692f451c03e7854243b95f4234b *tini" | sha256sum -c - && \
    mv tini /usr/local/bin/tini && \
    chmod +x /usr/local/bin/tini

ENTRYPOINT ["tini", "--"]
CMD ["jupyter", "notebook"]

### Add scripts and examples (needs work)
ADD files/run-ipython.sh /root/run-ipython.sh
ADD files/python /root/python
