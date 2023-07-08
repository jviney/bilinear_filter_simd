FROM ubuntu:23.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential cmake libopencv-dev
