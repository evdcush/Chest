#!/bin/bash

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#------------------------------------------------------------------------------
#                              Pre-setup stuff
#------------------------------------------------------------------------------
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

flameshot
boostnote
discord
zeal
joplin

#==============================================================================
#                         Package manager: apt-fast
#==============================================================================


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
# Pre-setup :
#    dirs and variables
#==============================================================================

# Make dirs
# ===============
mdp="mkdir -p "
mdph="$mdp $HOME/"
#mkdir -p $HOME/.apps
#mkdir -p $HOME/Projects/Clones-N-Forks
$mdph .apps
$mdph Projects/Clones-N-Forks


# Local script variables
#------------------------------------------------------------------------------

# Path vars
# ===============
APPSPATH="$HOME/.apps"
PROJPATH="$HOME/Projects"
CLONESPATH="$PROJPATH/Clones-N-Forks"

# git clone
# ===============
gclone_to='git clone --depth=1'

gclone(){
    cd $APPSPATH

}


# Functions
#------------------------------------------------------------------------------

# add apt ppa
# ===============
addrep(){
    add-apt-repository "ppa:$1" -y
    apt-fast update
}


#==============================================================================
#  Purge bloat and unwanted software
#==============================================================================

# Purge gnome shit
# ===============
# The ONLY package you want is 'gnome-themes-standard',
#  it would be easier to just wildcard gnome, and reinstall that...
$arp "gnome-*"
#$arp gnome-shell
#$arp gnome-accessibility-themes
#$arp gnome-bluetooth
#$arp gnome-calculator
#$arp "gnome-control*"
#$arp gnome-desktop3-data
#$arp gnome-keyring
#$arp gnome-menus
#$arp gnome-mines
#$arp gnome-sudoku
#$arp gnome-software-common
#$arp gnome-system-tools
#$arp gnome-user-guide
#$arp "gnome-icon*"
$arp "yelp*"
$aar # autoremove (going to be a lot)

# Purge ubuntu shit
# =================
#$arp "zenity"
$arp "xubuntu-desktop" # gets rid of basically EVERYTHING you don't want, including gnome shit
$aar

# purge bloat
# ===========
# NB: these tend to have extended supporting packages, so wildcard
$arp "*firefox*"
$arp "transmission-*"
$arp "*pidgin*"
$arp "*parole*"
$arp "*xfburn*"
#$arp evince # actually evince not horrible
$arp "*libreoffice*"
$arp "*mousepad*"
$aar


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#------------------------------------------------------------------------------
#                              Installation
#------------------------------------------------------------------------------
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Update, Upgrade
# ===============
$aug
update-ca-certificates -f
$aar



#==============================================================================
#                            Setup and Dependencies
#==============================================================================


#------------------------------------------------------------------------------
#                                  GCC
#------------------------------------------------------------------------------
# May not be necessary if we go 18.04??
addrep 'ubuntu-toolchain-r/test'
$ai "gcc-7 g++-7 gcc-8 g++-8 --install-suggests"
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 80 --slave /usr/bin/g++ g++ /usr/bin/g++-7
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 100 --slave /usr/bin/g++ g++ /usr/bin/g++-8
update-alternatives --set gcc /usr/bin/gcc-8
$aug


#------------------------------------------------------------------------------
#                           Dependencies & utils
#------------------------------------------------------------------------------
$ai curl
$ai wget
$ai apt-transport-https
$ai cmake
$ai make
$ai git-core
$ai autoconf
$ai automake
$ai build-essential
$ai libssl-dev
$ai zlib1g-dev
$ai libbz2-dev
$ai libreadline-dev
$ai libsqlite3-dev
$ai llvm
$ai libncurses5-dev
$ai xz-utils
$ai tk-dev
$ai libatomic1
$ai gfortran
$ai perl
$ai m4
$ai pkg-config
$ai software-properties-common
$ai p7zip-full
$ai p7zip-rar
$ai redis-server
$ai redis-tools
$ai qt5-default
$ai qt5-qmake
$ai qttools5-dev-tools
$ai libqt5dbus5 # just $ai "libqt5*" or?
$ai libqt5network5
$ai libqt5core5a
$ai libqt5widgets5
$ai libqt5gui5
$ai libqt5svg5-dev
$ai extra-cmake-modules
$ai libqt5webkit5-dev
$ai libqt5x11extras5-dev
$ai libarchive-dev
$ai libxcb-keysyms1-dev
$ai texlive-xetex
$au libcurl4-openssl-dev

# Printer utils
# ===============
$ai printer-driver-cups-pdf
$ai system-config-printer-gnome
$ai brother-cups-wrapper-common
$ai brother-cups-wrapper-extra
$ai brother-cups-wrapper-laser

#==============================================================================
#                                   Dev/env
#==============================================================================

# oh-my-zsh
# ===============
sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)";


# pyenv
# ===============
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash;
#echo 'export PATH="/home/evan/.pyenv/bin:$PATH"' > ~/.zshrc
#echo 'eval "$(pyenv init -)"' > ~/.zshrc
#echo 'eval "$(pyenv virtualenv-init -)"' > ~/.zshrc
# source ~/.zshrc
pyenv update
pyenv install 3.6.6
pyenv virtualenv 3.6.6 ^
pip install -U pip setuptools wheel
pip completion --zsh >> ~/.zshrc
source ~/.zshrc

# Node, NPM
# =================
curl -sL https://deb.nodesource.com/setup_10.x | -E bash -
$ai nodejs

# yarn
# =================
curl -sL https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list
$aug
$ai yarn


#==============================================================================
#                                   Software
#==============================================================================

#------------------------------------------------------------------------------
#                           package-manager packages
#------------------------------------------------------------------------------

# apt
# ===============
$ai alarm-clock-applet
$ai htop
$ai guake
$ai vim
$ai zsh
$ai tmux
$ai tree
$ai xfce4-timer-plugin
$ai anki
$ai chromium-browser
$ai psensor
$ai vlc
$ai mpv
$ai okular
$ai nomacs
$ai gparted
$ai tlp
$ai deluge
#$ai retext
%ai exfat-utils
$ai flac
$ai lame
$ai x264
$ai unrar
#$ai ubuntu-restricted-extras
#$arp flashplugin-installer
$ai dictd
$ai dict-gcide
$ai dict-wn


# pip
# ===============
pi="pip install"
$pi numpy
$pi scipy
$pi sklearn
$pi matplotlib
$pi ipython
$pi jupyter
$pi jupyterthemes
$pi chainer
$pi chainerrl
$pi apt-select # !!!!!!!!!!
#$pi atari-py
#$pi attrs
$pi bs4
$pi bibtexparser
$pi cookiecutter # !!!!!!!!!!!!!
$pi gym
$pi h5py
$pi hickle
$pi html5lib
$pi neupy
$pi nose
$pi pandas
$pi pandocfilters
$pi Pillow
$pi pygame
#$pi PySC2
$pi pyqt5
$pi rebound-cli # !!!!!!
$pi s2clientprotocol
$pi sc2
#$pi scbw
$pi thefuck # !!!!!!!!!!
#$pi pptree
#$pi pipdeptree
#$pi pprintpp
$pi afdko # adobe font development kit, ligatures
$pi pip-check # check package versions and list outdated
$pi starred # !!!! : generates starred
$pi termdown # terminal countdown timer/stopwatch
$pi cairosvg # converts svg to pdf or png
$pi python-qtpip # gui manager for pip
$pi grip # preview github markdown before uploading
$pi gitsuggest # suggest repos based on your stars
$pi art # ascii art tool (text -> ascii)
$pi betterbib # !!!!!!!!!!!!!!! can fix bibtext from info retrieved
$pi pyment # !!! format and convert python docstrings
$pi docutils # rst source
$pi rst # Module to create reStructuredText documents through code.
#$pi rstvalidator
#$pi reflowrst # Modify valid rst text to fit within given width
$pi retext # rst editor with live preview (kind of sucks)
$pi rstwatch # !!!!!!!!!!!!!!!!!!!! Watch directories for changes to RST files and generate HTML
$pi rstdiary # Create static HTML diary from single RST input
$pi rstvalidator
$pi rst-archiver # Add files to your RST notes more easily.
$pi sphinx # EVERYTHING
$pi sphinxcontrib-jupyter # Sphinx "Jupyter" extension: Convert your RST files into executable Jupyter notebooks.
$pi pelican # !!!!!!! sphinx for generating articles blog
$pi pelican_publications # ????? !!!! A Pelican plugin that adds an RST directive for including a BibTeX publication list.
$pi oh-my-stars # search your github stars locally cli

# npm ?
# ===============
npm install nativefier -g
https://svgporn.com/

#------------------------------------------------------------------------------
#                             ppa/sources.list packages
#------------------------------------------------------------------------------

# Ghostwriter
# ===============
addrep wereturtle/ppa
$ai ghostwriter

# Font manager
# ===============
addrep font-manager/staging
$ai font-manager

# Paper icons
# ===============
addrep snwh/ppa
$ai paper-icon-theme

# Sources.list
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Sublime Text 3
# ===============
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | apt-key add -;
echo "deb https://download.sublimetext.com/ apt/stable/" | tee /etc/apt/sources.list.d/sublime-text.list;
$au
$ai sublime-text

# Super productivity
# ===============
echo "deb https://dl.bintray.com/johannesjo/super-productivity stable super-productivity" | tee -a /etc/apt/sources.list
$aug
$ai superproductivity





#------------------------------------------------------------------------------
#                            packages with installers
#------------------------------------------------------------------------------
# oh-my-zsh, pyenv both above

# fzf
# ===============
  # A command line fuzzy finder
$gclone https://github.com/junegunn/fzf.git "$APPSPATH/fzf" && cd "$APPSPATH/fzf"
$"bash install"
cd
# Joplin
# ===============
wget -O - https://raw.githubusercontent.com/laurent22/joplin/master/install_ubuntu.sh | bash


# z
# ===============
# Jump around
$gclone https://github.com/rupa/z.git "$APPSPATH/z_jump"
@TODO

#------------------------------------------------------------------------------
#                                build from source
#------------------------------------------------------------------------------

# Flameshot
# ===============
$gclone https://github.com/lupoDharkael/flameshot "$APPSPATH/flameshot" && cd "$APPSPATH/flameshot"
mkdir build && cd build
qmake ../
make
make install # sudo make install
cd



#                       ______                                                 #
#                      /\     \                                                #
#                     /  \     \                                               #
#                    /    \_____\                                              #
#                   _\    / ____/_                                             #
#                  /\ \  / /\     \                                            #
#                 /  \ \/_/  \     \                                           #
#                /    \__/    \_____\                                          #
#               _\    /  \    / ____/_                                         #
#              /\ \  /    \  / /\     \                                        #
#             /  \ \/_____/\/_/  \     \                                       #
#            /    \_____\    /    \_____\                                      #
#           _\    /     /    \    / ____/_                                     #
#          /\ \  /     /      \  / /\     \                                    #
#         /  \ \/_____/        \/_/  \     \                                   #
#        /    \_____\            /    \_____\                                  #
#       _\    /     /            \    / ____/_                                 #
#      /\ \  /     /              \  / /\     \                                #
#     /  \ \/_____/                \/_/  \     \                               #
#    /    \_____\                    /    \_____\                              #
#   _\    /     /_  ______  ______  _\____/ ____/_                             #
#  /\ \  /     /  \/\     \/\     \/\     \/\     \                            #
# /  \ \/_____/    \ \     \ \     \ \     \ \     \                           #
#/    \_____\ \_____\ \_____\ \_____\ \_____\ \_____\                          #
#\    /     / /     / /     / /     / /     / /     /                          #
# \  /     / /     / /     / /     / /     / /     /                           #
#  \/_____/\/_____/\/_____/\/_____/\/_____/\/_____/                            #

                USE THE GCD FUNC IN ZSHRC!
# Zeal
# ===============
$gclone https://github.com/zealdocs/zeal "$APPSPATH/zeal"''
$gclone https://github.com/lupoDharkael/flameshot "$APPSPATH/flameshot" && cd "$APPSPATH/flameshot"
mkdir build && cd build
qmake ../
make
make install # sudo make install
cd




nativefier "https://www.hackerrank.com/dashboard" -n hacker-rank -a x64 -p linux --title-bar-style hidden --honest -m -i ./Icons/HackerRank_logo.png
