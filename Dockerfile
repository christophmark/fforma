FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       locales \
       build-essential \
       r-base \
       r-base-dev \
       python3 \
       python3-pip \
       python3-setuptools  \
       python3-dev \
       git \
       vim \
    && pip3 --no-cache-dir install --upgrade \
       setuptools \
       wheel

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_TYPE en_US.UTF-8

RUN pip3 install -r requirements.txt

RUN apt-get install -y libcurl4-openssl-dev
RUN Rscript -e "install.packages('forecast')"

RUN mkdir /app/data

COPY . /app
