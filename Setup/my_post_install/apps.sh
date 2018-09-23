#!/bin/bash

#==============================================================================
#------------------------------------------------------------------------------
#                              Pre-setup stuff
#------------------------------------------------------------------------------
#==============================================================================

#==============================================================================
# Pre-setup setup :
#    dirs and variables
#==============================================================================

# Make dirs
# ===============
mkdir -p $HOME/.apps
mkdir -p $HOME/Projects/Clones-N-Forks

#------------------------------------------------------------------------------
# Local script variables
#------------------------------------------------------------------------------

# Path vars
# ===============
APPSPATH="$HOME/.apps"
PROJPATH="$HOME/Projects"
CLONESPATH="$PROJPATH/Clones-N-Forks"

# git clone
# ===============
gclone='git clone --depth=1'



#==============================================================================
# Pre-setup package manager:
#   apt-fast
#==============================================================================

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------
# Normally apt-fast prompts the user for configuration of a few
#  settings. So not sure if it needs to be done on it's own
#  or whether this "one-liner" installation can bypass all that

#===== Apparently a complete one-liner
/bin/bash -c "$(curl -sL https://git.io/vokNn)"

#===== How I did it before
#sudo add-apt-repository ppa:apt-fast/stable
#sudo apt update
#sudo apt install apt-fast -y

# apt-fast mirrors
# ===============
sed -i "$ a MIRRORS=( 'http://mirror.picosecond.org/ubuntu/', 'http://ftp://mirror.picosecond.org/ubuntu/', 'http://rsync://mirror.picosecond.org/ubuntu/', 'http://ftp://mirror.enzu.com/ubuntu/', 'http://rsync://mirror.enzu.com/ubuntu/', 'http://mirror.enzu.com/ubuntu/', 'http://la-mirrors.evowise.com/ubuntu/', 'http://rsync://mirrors.ocf.berkeley.edu/ubuntu/', 'http://ftp://mirrors.ocf.berkeley.edu/ubuntu/', 'http://mirrors.ocf.berkeley.edu/ubuntu/', 'http://mirror.math.ucdavis.edu/ubuntu/' )" /etc/apt-fast.conf

#------------------------------------------------------------------------------
# Setting up local vars for apt-fast
#------------------------------------------------------------------------------
# definition : varname='my command'
#      usage : $varname thing

#----- apt-fast
a="apt-fast"

#----- apt-fast remove
arp="$a remove --purge -y"
aar="$a autoremove -y"

#----- apt-fast updating
au="$a update"
ag="$a upgrade -y"
aug="$au && $ag"

#----- apt-fast install
ai="$a install -y"


#==============================================================================
#  Purge bloat and unwanted software
#==============================================================================

# Purge gnome shit
# ===============
$arp gnome-shell
$arp gnome-accessibility-themes
$arp gnome-bluetooth
$arp gnome-calculator
$arp "gnome-control*"
$arp gnome-desktop3-data
$arp gnome-keyring
$arp gnome-menus
$arp gnome-mines
$arp gnome-sudoku
$arp gnome-software-common
$arp gnome-system-tools
$arp gnome-user-guide
$arp "gnome-icon*"
$arp "yelp*"

$aar

# Purge ubuntu shit
# ===============
$arp "zenity*"

$aar

# purge bloat
# ===============
$arp firefox
$arp transmission-common
$arp pidgin
$arp pidgin-data
$arp pidgin-otr
$arp parole
$arp xfburn
$arp evince
$arp libreoffice-common
$arp mousepad

$aar

#==============================================================================
#------------------------------------------------------------------------------
#                              Setup for stuff
#------------------------------------------------------------------------------
#==============================================================================

# Update, Upgrade
# ===============
$aug
update-ca-certificates -f
$aar
# apt full-upgrade # for drivers, but may install ubuntu-sourced nvidia


#==============================================================================
#  Apt packages installation
#==============================================================================

#------------------------------------------------------------------------------
#                              Dev/env/utils
#------------------------------------------------------------------------------

# Libraries & Dependencies
# ===============



#------------------------------------------------------------------------------
#                              Software
#------------------------------------------------------------------------------

# Apps
# ===============



# Appearance
# ===============


# Utility
# ===============
$ai alarm-clock-applet



#==============================================================================
#  Non-apt package/software installation
#==============================================================================

#------------------------------------------------------------------------------
# Shell & CLI utils
#------------------------------------------------------------------------------

# oh-my-zsh
# ===============
sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)";

# fzf
# ===============
  # A command line fuzzy finder
$gclone https://github.com/junegunn/fzf.git "$APPSPATH/fzf" && cd "$APPSPATH/fzf"
bash install

# z
# ===============
# Jump around
$gclone https://github.com/rupa/z.git "$APPSPATH/z_jump"

#------------------------------------------------------------------------------
# Software
#------------------------------------------------------------------------------
