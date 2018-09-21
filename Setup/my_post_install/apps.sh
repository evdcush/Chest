#!/bin/bash

APPSPATH='/home/evan/.apps'
mkdir -p /home/evan/.apps

#==============================================================================
#  Utils
#==============================================================================



#------------------------------------------------------------------------------
# CLI utils
#------------------------------------------------------------------------------

APPS=

# fzf
# ===============
  # A command line fuzzy finder
git clone --depth=1 git@github.com:junegunn/fzf.git ~/.apps/ && cd ~/.apps/fzf
bash install


# z
# ===============
  # Jump around
#git clone https://github.com/rupa/z.git ~/Soft/Installed/Utils/z_jump
