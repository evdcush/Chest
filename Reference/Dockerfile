# Pull focal
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

# Update packages.
RUN apt update && apt upgrade -y

# Install packages.
RUN apt install -y \
    git curl wget tree htop zsh python3-pip mpich
