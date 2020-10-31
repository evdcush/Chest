# Pull
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

# Update packages.
RUN apt update && apt upgrade -y

# Install packages.
RUN apt install -y \
    git curl wget tree htop zsh make automake cmake autoconf m4 build-essential \
    gcc-9 g++-9 llvm mpich gfortran libboost-all-dev libblas-dev liblapack-dev \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev \
    libjpeg-dev liblzma-dev

