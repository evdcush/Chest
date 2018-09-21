#!/bin/bash


mkdir -p $HOME/.apps
mkdir -p $HOME/Projects

# Paths
APPSPATH="$HOME/.apps/"
PROJPATH="$HOME/Projects/"

#==============================================================================
#  PRE-ANYTHING
#==============================================================================

# purge bloat
# ===============
apt remove --purge -y firefox
apt remove --purge -y transmission-common
apt remove --purge -y pidgin
apt remove --purge -y pidgin-data
apt remove --purge -y pidgin-otr
apt remove --purge -y parole
apt remove --purge -y xfburn
apt remove --purge -y evince
apt remove --purge -y libreoffice-common
apt remove --purge -y mousepad


# Update, Upgrade
# ===============
#sudo apt update && sudo apt upgrade -y && sudo update-ca-certificates -f && sudo apt autoremove
apt update
apt upgrade -y
update-ca-certificates -f
apt autoremove -y
# apt full-upgrade # for drivers, but may install ubuntu-sourced nvidia


# apt-fast
# ===============
/bin/bash -c "$(curl -sL https://git.io/vokNn)"
#sudo add-apt-repository ppa:apt-fast/stable
#sudo apt update
#sudo apt install apt-fast -y
sed -i '$ a "MIRRORS=( 'http://mirror.picosecond.org/ubuntu/', 'http://ftp://mirror.picosecond.org/ubuntu/', 'http://rsync://mirror.picosecond.org/ubuntu/', 'http://ftp://mirror.enzu.com/ubuntu/', 'http://rsync://mirror.enzu.com/ubuntu/', 'http://mirror.enzu.com/ubuntu/', 'http://la-mirrors.evowise.com/ubuntu/', 'http://rsync://mirrors.ocf.berkeley.edu/ubuntu/', 'http://ftp://mirrors.ocf.berkeley.edu/ubuntu/', 'http://mirrors.ocf.berkeley.edu/ubuntu/', 'http://mirror.math.ucdavis.edu/ubuntu/' )"' /etc/apt-fast.conf
# THIS WONT WORK, MAKE A FILE, TEE IT OR WHATEVER
#==============================================================================
#  Utils
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
git clone --depth=1 https://github.com/junegunn/fzf.git APPSPATH && cd "$APPSPATH/fzf"
bash install


# z
# ===============
  # Jump around
#git clone https://github.com/rupa/z.git ~/Soft/Installed/Utils/z_jump

