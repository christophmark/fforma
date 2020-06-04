FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive
ENV LANG en_US.UTF-8
ENV LC_TYPE en_US.UTF-8

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       r-base \
       python3.7 \
       python3-pip \
       python3-setuptools  \
       python3-dev \
       git \
    && pip3 --no-cache-dir install --upgrade \
       setuptools \
       wheel

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt

RUN Rscript -e "install.packages('forecast')"
RUN Rscript -e "install.packages('tsfeatures')"

COPY . /app
