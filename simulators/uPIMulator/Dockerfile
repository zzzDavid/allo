FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git wget cmake ninja-build pkg-config \
    python3 python3-pip libc6 \
    && rm -rf /var/lib/apt/lists/*

# UPMEM SDK 2023.1.0 (from CMU-SAFARI/UPMEM mirror; official host returns 403)
WORKDIR /root
COPY upmem-sdk.tar.gz /root/upmem-sdk.tar.gz
RUN tar -zxf upmem-sdk.tar.gz && rm upmem-sdk.tar.gz

ENV UPMEM_HOME=/root/upmem-2023.1.0-Linux-x86_64
ENV PATH=${UPMEM_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${UPMEM_HOME}/lib
ENV PYTHONPATH=${UPMEM_HOME}/lib/python3/dist-packages

WORKDIR /root/uPIMulator
