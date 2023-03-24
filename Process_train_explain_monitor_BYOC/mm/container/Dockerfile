# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:18.04

MAINTAINER Amazon AI <sage-learner@amazon.com>
# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.

RUN apt-get update \
 && apt-get install wget unzip zip -y \
 && apt-get install make


#RUN apt-get install gcc
RUN apt-get install -y build-essential
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN wget https://bootstrap.pypa.io/pip/3.3/get-pip.py 

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip3 install wheel
RUN pip install scikit-build
RUN pip install cmake==3.22.6
RUN pip install requests
RUN pip --no-cache-dir install numpy==1.16.2 scipy==1.2.1 scikit-learn==0.22.0 pandas flask gunicorn
RUN pip install --pre xgboost==0.90

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY xgboost /opt/program
WORKDIR /opt/program

