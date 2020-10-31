# Pull image.
# FROM ubuntu:20.04
# FROM alpine:latest
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

# Set timezone.
ENV TZ=Asia/Tokyo
RUN ln -sfn /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Update packages.
RUN apt update && apt upgrade -y

# Install packages.
RUN apt install -y \
    git curl wget tree htop zsh make automake cmake autoconf m4 build-essential \
    gcc-9 g++-9 llvm mpich gfortran libboost-all-dev libblas-dev liblapack-dev \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev locales \
    libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev \
    software-properties-common libjpeg-dev liblzma-dev jq yq vim \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/archives/*

# Set locale
RUN locale-gen en_US.UTF-8

# @TODO Setup zsh
#   --> I think you have to setup $USER and .zshrc first??
# RUN chsh -s $(which zsh)
# RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# @TODO Install pyenv
#   --> I think you need to have done the .zshrc first
#RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | zsh



#=============================================================================#
#                                  REFERENCE                                  #
#=============================================================================#

# https://github.com/topics/zsh?l=dockerfile
# https://github.com/topics/oh-my-zsh?l=dockerfile
# https://github.com/topics/pytorch?l=dockerfile