#!/bin/bash

##  Script for installing Julia from tarball  ##


# Version
# -------
major='1.3'
minor='1'
VER="$major.$minor"

# Formatting
# ----------
url="https://julialang-s3.julialang.org/bin/linux/x64/$major/julia-$VER-linux-x86_64.tar.gz"
tball="julia-$VER-linux-x86_64.tar.gz"
jdir="julia-$VER"


# Get & install
# =============
wget $url
tar -xvf $tball
sudo mv $jdir /opt
sudo ln -sf /opt/$jdir/bin/julia /usr/local/bin

