#!/bin/bash


# REQUIREMENTS:
# python-pip:
#   - sphinx
#   - doc2dash
pip install -U doc2dash # installs sphix as dep

#clone chainer repo:
git clone git@github.com:chainer/chainer.git && cd chainer/docs

# my sphinx-build failed on master
# so I switched to 'doc-fusion', and it worked
git checkout doc-fusion && make html

#Get the icon image from http://chainer.org/images/chainer_icon_red.png
doc2dash -n Chainer -i chainer_icon_red.png -j build/html
