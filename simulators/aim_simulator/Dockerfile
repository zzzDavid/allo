FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    g++-12 cmake make git python3 \
    && rm -rf /var/lib/apt/lists/*

ENV CXX=g++-12
ENV CC=gcc-12

WORKDIR /work
