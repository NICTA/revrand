# Docker file for uncoverml project

FROM ubuntu:16.04
MAINTAINER Daniel Steinberg <daniel.steinberg@data61.csiro.au>

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8
RUN apt-get update && apt-get install -y \
  build-essential \
  # Fast blas for numpy and scipy
  libopenblas-base \ 
  libopenblas-dev \
  python3 \
  python3-dev\
  python3-pip \
  # Clean up
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  # Make folders
  && mkdir -p /usr/src/python/revrand

# pip packages 
RUN pip3 -v install \
  Cython \
  numpy \
  scipy \
  scikit-learn \
  unipath \
  pytest \
  pytest-cov \
  codecov

WORKDIR /usr/src/python/revrand

COPY . /usr/src/python/revrand

# Install this package
RUN python3 setup.py develop
