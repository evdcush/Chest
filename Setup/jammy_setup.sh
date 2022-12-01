exit;  # Safety-first!
#=============================================================================#
#                                                                             #
#       gg                                                                    #
#      dP8,                                                                   #
#     dP Yb                                                                   #
#    ,8  `8,                                                                  #
#    I8   Yb                                                                  #
#    `8b, `8,     ,gggg,gg   ,ggg,,ggg,,ggg,    ,ggg,,ggg,,ggg,   gg     gg   #
#     `"Y88888   dP"  "Y8I  ,8" "8P" "8P" "8,  ,8" "8P" "8P" "8,  I8     8I   #
#         "Y8   i8'    ,8I  I8   8I   8I   8I  I8   8I   8I   8I  I8,   ,8I   #
#          ,88,,d8,   ,d8b,,dP   8I   8I   Yb,,dP   8I   8I   Yb,,d8b, ,d8I   #
#      ,ad88888P"Y8888P"`Y88P'   8I   8I   `Y88P'   8I   8I   `Y8P""Y88P"888  #
#    ,dP"'   Yb                                                        ,d8I'  #
#   ,8'      I8                                                      ,dP'8I   #
#  ,8'       I8                    ██████████                       ,8"  8I   #
#  I8,      ,8'                ████░░░░░░░░░░████                   I8   8I   #
#  `Y8,___,d8'             ████░░░░░░░░░░░░░░░░░░████               `8, ,8I   #
#    "Y888P"             ██░░░░░░░░░░░░░░░░░░░░░░░░░░██              `Y8P"    #
#                      ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██                     #
#                      ██░░░░░░░░██░░░░░░░░░░██░░░░░░░░██                     #
#                    ██░░░░░░░░░░██░░░░░░░░░░██░░░░░░░░░░██                   #
#                    ██░░░░▒▒▒▒░░░░░░░░░░░░░░░░░░▒▒▒▒░░░░██                   #
#                    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██                   #
#                      ██░░░░░░░░██░░░░░░░░░░██░░░░░░░░██                     #
#                        ████████  ██████████  ████████                       #
#                          ██          ██          ██                         #
#                        ██░░██      ██░░██      ██░░██                       #
#                        ██░░░░██  ██░░░░██      ██░░░░██                     #
#                          ██░░██  ██░░██          ██░░██                     #
#                        ██░░░░██  ██░░░░██      ██░░░░██                     #
#                        ██░░██      ██░░██      ██░░██                       #
#                          ██          ██          ██                         #
#                          ██          ██          ██                         #
#                        ██░░██      ██░░██      ██░░██                       #
#                      ██░░░░██      ██░░░░██  ██░░░░██                       #
#                      ██░░██          ██░░██  ██░░██                         #
#                      ██░░░░██      ██░░░░██  ██░░░░██                       #
#                        ██░░██      ██░░██      ██░░██                       #
#                          ██          ██          ██                         #
#                                                                             #
#=============================================================================#

# BEFORE YOU ACCIDENTLY BUILD THIS SCRIPT BECAUSE SUBLIME TEXT, IN THEIR
# INFINITE BRILLIANCE, DECIDED TO KEYBIND "build" to ctrl+b, AS IF IT WERE NOT
# LITERALLY RIGHT NEXT TO THE THE MOST COMMONLY USED KEYBINDING EVER (ctrl+v).
# DISABLE THAT BULLSHIT:
#[
#    { "keys": ["ctrl+k", "ctrl+q"], "command": "toggle_setting", "args": {"setting": "word_wrap"} },
#    { "keys": ["f7"], "command": "noop" },
#    { "keys": ["ctrl+b"], "command": "noop" },
#    { "keys": ["ctrl+shift+b"], "command": "noop", "args": {"select": true} },
#    { "keys": ["ctrl+break"], "command": "noop" },
#]


# Get the image.
wget https://ftp.riken.jp/Linux/ubuntu-releases/jammy/ubuntu-22.04.1-desktop-amd64.iso

# Burn it to a USB using Etcher.
#   NB: don't be a dd hero here; it's either etcher or pain. Version doesn't matter.
wget https://github.com/balena-io/etcher/releases/download/v1.7.9/balenaEtcher-1.7.9-x64.AppImage -O etcher && chmod +x etcher
./etcher

#=============================================================================#
#                                                                             #
#                         ███████ ███    ██ ██    ██                          #
#                         ██      ████   ██ ██    ██                          #
#                         █████   ██ ██  ██ ██    ██                          #
#                         ██      ██  ██ ██  ██  ██                           #
#                         ███████ ██   ████   ████                            #
#                                                                             #
#=============================================================================#

# Remove bloat
# ============
sudo apt remove --purge firefox snapd
sudo apt autoremove

# 𝗗𝗜𝗦𝗔𝗕𝗟𝗘 𝗖𝗔𝗡𝗢𝗡𝗜𝗖𝗔𝗟 𝗠𝗔𝗥𝗞𝗘𝗧𝗜𝗡𝗚
# https://askubuntu.com/questions/1434512/how-to-get-rid-of-ubuntu-pro-advertisement-when-updating-apt
sudo rm /etc/update-motd.d/88-esm-announce
sudo systemctl disable ubuntu-advantage

# """OFFICIAL""" solution from canonical.
sudo pro config set apt_news=false


sudo vi /etc/default/motd-news
#---> change `ENABLED=1` to `ENABLED=0`

#-----------------------------------------------------------------------------#
#                           ___   _   _   ___      _                          #
#                          / __| | | | | |   \    /_\                         #
#                         | (__  | |_| | | |) |  / _ \                        #
#                          \___|  \___/  |___/  /_/ \_\                       #
#                                                                             #
#-----------------------------------------------------------------------------#
# First-things-first, CUDA. If CUDA is fine, everything else is manageable.
# (USING VERSIONS AT THE TIME OF INSTALLATION)

# Get the network installer deb.
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb

# (Remove any existing cuda keys that may have been installed by default buntu 3rd party setup):
sudo apt-key del 7fa2af80

# Install CUDA keys.
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Install CUDA.
#   Note: I'm not using pinned versions; I don't need to. If you do, just install that ver.
sudo apt update && sudo apt install -y cuda

# CUDA SDK packages (after reboot!).
sudo apt install -y libcudnn8

###############################################################################


#=============================================================================#
#                                     Deps                                    #
#=============================================================================#

# Generally needed.
sudo apt install -y curl git git-crypt ffmpeg imagemagick python3-pip ssh xclip vim zsh

# Real shit.
sudo apt install libboost-all-dev mpich swig
sudo apt install gcc-12 g++-12
# Can optionally setup alternatives:
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# Extras.
sudo apt install -y flac git-extras lame x264 x265

# Python (pyenv)
# ==============
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Snowflakes
# ==========
#=== AppImage
# Due to some AppImage dep req on fuse (2), see: https://github.com/AppImage/AppImageKit/issues/1120
sudo apt install -y libfuse2

#=============================================================================#
#                                 Environment                                 #
#=============================================================================#
# These steps all assume you have installed the deps above.

# PATH
# ====
# Default: 
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"

# Changed:
## (no snap, home bin)
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/home/evan/.local/bin"

# Oh-my-zsh
# =========
sudo apt install -y git zsh && sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
# (It's worth starting a new sess after install)
# Install zsh plugins:
git clone --depth=1 https://github.com/jocelynmallon/zshmarks.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zshmarks
git clone --depth=1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# ==== #
# 𝗗𝗢𝗧𝗦 #
# ==== #
ln -sf $HOME/Chest/Dots $HOME/.Dots

#█████████████████████████████████████████████████████████████████████████████#
#                                     GIT                                     #
#█████████████████████████████████████████████████████████████████████████████#


# Git
# ===
# You can symlink your Dots/gitconfig to ~/.gitconfig
ln -sf $HOME/.Dots/gitconfig $HOME/.gitconfig
# You should definitely symlink your global gitignore:
ln -sf $HOME/.Dots/global_gitignore $HOME/.gitignore

# Or specify options manually:
git config --global user.name "Evan C."
git config --global user.email "evdcush@protonmail.com"
git config --global init.defaultBranch master
git config --global core.excludesFile '~/.gitignore'
git config --global core.pager ''
git config --global help.autocorrect 8  # 0.8s to cancel an auto-corrected command.
git config --global url.'git@github.com:'.insteadOf 'https://github.com/'  # use ssh instead of https for remotes

# After npm i -g git-branch-select:
git config --global alias.bs branch-select

# Figure out which signing key you want:
gpg --list-secret-keys --keyid-format=long
# Set it:
git config --local user.signingkey 0123456789ABCDEF
git config --global commit.gpgsign true

#=============================================================================#

# fzf
# ===
# Install fzf:
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install

# pyenv
# =====
# (Assumes you installed deps above!)
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install pyenv.
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

# Add the following path stuff to the shell config file (if not already there).
export PYENV_ROOT="$HOME/.pyenv"
export PATH="${PATH:+${PATH}:}$PYENV_ROOT/bin"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Source shell file.
exec $SHELL

# Install python version, eg:
pyenv install 3.10.6

# Make venv.
pyenv virtualenv 3.10.6 3106


#█████████████████████████████████████████████████████████████████████████████#
#                                  BLUETOOTH                                  #
#█████████████████████████████████████████████████████████████████████████████#

# BLUEMAN
# =======
# How to get the latest version.

# First, get whatever is the latest the deb package available for the
# most current/newest version of Blueman is.
# This will likely be the package distributed to the latest release of Ubuntu.
# You can find the latest version HERE:
https://packages.ubuntu.com/search?suite=all&arch=amd64&searchon=names&keywords=blueman

# At the time of writing this note, the latest package is published in the
# latest release of Ubuntu:
#   Blueman: 2.3.2-1
#   Kinetic(x11)

# Now, go to the page for that package release:
https://packages.ubuntu.com/kinetic/blueman

# Find the download page below:
https://packages.ubuntu.com/kinetic/amd64/blueman/download
# 'Download Page for blueman_2.3.2-1_amd64.deb on AMD64 machines'

# Download it from whatever mirror you want (I picked what is probably the closest to me):
wget http://kr.archive.ubuntu.com/ubuntu/pool/universe/b/blueman/blueman_2.3.2-1_amd64.deb

#███████  UNSATISFIED/CONFLICTING DEPENDENCY: it needs latest python!  ███████#

# $ sudo dpkg -i blueman_2.3.2-1_amd64.deb
# (Reading database ... 366200 files and directories currently installed.)
# Preparing to unpack blueman_2.3.2-1_amd64.deb ...
# Unpacking blueman (2.3.2-1) over (2.1.2-1ubuntu0.3) ...
# dpkg: dependency problems prevent configuration of blueman:
#  blueman depends on python3 (>= 3.10~); however:
#   Version of python3 on system is 3.8.2-0ubuntu2.
#
# dpkg: error processing package blueman (--install):
#  dependency problems - leaving unconfigured
# Processing triggers for gnome-menus (3.36.0-1ubuntu1) ...
# Processing triggers for desktop-file-utils (0.24-1ubuntu3) ...
# Processing triggers for mime-support (3.64ubuntu1) ...
# Processing triggers for dbus (1.12.16-2ubuntu2.3) ...
# Processing triggers for libglib2.0-0:amd64 (2.64.6-1~ubuntu20.04.4) ...
# Processing triggers for hicolor-icon-theme (0.17-2) ...
# Processing triggers for man-db (2.9.1-1) ...
# Errors were encountered while processing:
#  blueman


# Okay, let's install that version of Python then!
# We'll of course need the deadsnakes ppa for this:
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update



#



#=============================================================================#
#                               Settings/Config                               #
#=============================================================================#

# Gnome stuff
# ===========
sudo apt install -y gnome-tweaks nautilus-actions

#=== Extensions
# SYSTEM-MONITOR: https://extensions.gnome.org/extension/3010/system-monitor-next/
#                 https://github.com/mgalgs/gnome-shell-system-monitor-applet


# SSH key
# =======
sudo apt install -y ssh && ssh-keygen -t ed25519 -C 'evdcush@protonmail.com'
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519
xclip ~/.ssh/id_ed25519.pub


# Japanese
# ========
# ⚠️ WIP, STILL NOT WORKING AS EXPECTED ⚠️
#   - my install workflow for bionic & focal (mozc + pray) didn't work here
#   - it's stuck on that pleb ibus "direct-input" shit 🆖
#   - NB: LOOK INTO ``fcitx5`` ????
#     - why the F does it need to be like this
#     - literally never have switched input methods to JA so that I could type in "direct-input"
#     - figure out what peeps doing for jammy on US-layout keyboards on qita


#=============================================================================#
#                                    Gnome                                    #
#=============================================================================#

# SHELL EXTENSIONS
# ================
# Essential:
sudo apt install -y gnome-shell-extensions \
gnome-shell-extension-dash-to-panel \
gnome-shell-extension-no-annoyance

# Others:
sudo apt install -y gnome-shell-extensions \
gnome-shell-extension-arc-menu \
gnome-shell-extension-weather \
gnome-shell-extension-system-monitor \
gnome-shell-pomodoro \
gnome-shell-timer

# Maybe?
# gnome-shell-extension-gsconnect \
# gnome-shell-extension-move-clock \



#=============================================================================#
#                                     Misc                                    #
#=============================================================================#

#=============================================================================#
#                                                                             #
#                      ██████  ██    ██  ██████  ███████                      #
#                      ██   ██ ██    ██ ██       ██                           #
#                      ██████  ██    ██ ██   ███ ███████                      #
#                      ██   ██ ██    ██ ██    ██      ██                      #
#                      ██████   ██████   ██████  ███████                      #
#                                                                             #
#=============================================================================#


#  CHROMIUM-based BROWSERS FREEZING ON 22.04
#   Ref:
#     - https://askubuntu.com/a/1406495
#   Description:
#     After downloading something, that browser window freezes. Other browser
#     windows are still functional, but the window in which the download
#     completed is frozen and will stay frozen until killed.
sudo apt install xdg-desktop-portal-gnome

#=============================================================================#
#                                                                             #
#                        █████  ██████  ██████  ███████                       #
#                       ██   ██ ██   ██ ██   ██ ██                            #
#                       ███████ ██████  ██████  ███████                       #
#                       ██   ██ ██      ██           ██                       #
#                       ██   ██ ██      ██      ███████                       #
#                                                                             #
#=============================================================================#

# General
# =======
sudo apt install -y copyq deluge flameshot gparted inkscape keepassxc nextcloud-desktop nomacs pandoc redshift wkhtmltopdf

#=== Misc CLI tools
# currency conversion
sudo apt install -y qalc

#=== Terminal (GUAKE)
# RESOURCES:
#   https://wiki.archlinux.org/title/Guake
#   https://guake.readthedocs.io/en/latest/index.html
#   https://github.com/Guake/guake
#
# Don't install from official apt sources; they're always outdated.
# Add the PPA first, then install.
sudo add-apt-repository -y ppa:linuxuprising/guake
sudo apt update

sudo apt install -y guake

# (Patch guake if `Could not parse file "/usr/share/applications/guake.desktop": No such file or directory`)
sudo ln -sf /usr/share/guake/autostart-guake.desktop /usr/share/applications/guake.desktop

#=== Devtools
# GIT
# ---
# Enhancements
sudo apt install -y git-extras git-sizer git-quick-stats git-annex

# GUI
sudo apt install -y meld

#== Download stuff
sudo apt install -y aria2
# croc
curl https://getcroc.schollz.com | bash



#== CLI.
#sudo apt install -y aria2 bat delta duf jq neofetch screenfetch tree
sudo apt install -y aria2 delta duf jq neofetch screenfetch tree

## BAT MAY BE OLD through official apt, can download directly, eg:

#-----------------------------------------------------------------------------#
#                               GH Release Apps                               #
#-----------------------------------------------------------------------------#

#=== bat
# https://github.com/sharkdp/bat
## eg: 'bat_0.22.1_amd64.deb'
curl https://api.github.com/repos/sharkdp/bat/releases/latest | \
jq '.assets[] | select(.name | startswith("bat_") and endswith("_amd64.deb")) | .browser_download_url' | \
xargs wget -O bat.deb && \
sudo dpkg -i bat.deb \
&& rm bat.deb;

#=== flameshot
# https://github.com/flameshot-org/flameshot
## eg:
##   - 'flameshot-12.1.0-1.ubuntu-20.04.amd64.deb'
##   - 'flameshot-12.1.0-1.ubuntu-22.04.amd64.deb'
curl https://api.github.com/repos/flameshot-org/flameshot/releases/latest | \
jq '.assets[] | select(.name | startswith("flameshot-") and endswith(".ubuntu-20.04.amd64.deb")) | .browser_download_url' | \
xargs wget -O flameshot.deb && \
sudo dpkg -i flameshot.deb \
&& rm flameshot.deb;

#=== duf
# https://github.com/muesli/duf
## eg: 'duf_0.8.1_linux_amd64.deb'
curl https://api.github.com/repos/muesli/duf/releases/latest | \
jq '.assets[] | select(.name | startswith("duf_") and endswith("_linux_amd64.deb")) | .browser_download_url' | \
xargs wget -O duf.deb && \
sudo dpkg -i duf.deb \
&& rm duf.deb;



#=== Top.
sudo apt install -y htop
wget https://github.com/ClementTsang/bottom/releases/download/0.6.8/bottom_0.6.8_amd64.deb && sudo dpkg -i bottom_0.6.8_amd64.deb

#== Media.
sudo apt install -y catimg mpv sox vlc mkvtoolnix webp

#== Archive/FS
sudo apt install -y exfat-utils p7zip-full unar

#== PDF stuff.
sudo apt install -y texlive-xetex texlive-extra-utils

#== Dict.
sudo apt install -y dict dictd dict-gcide dict-wn
sudo apt install -y dict-freedict-eng-jpn dict-freedict-eng-lat
# `dict-moby-thesaurus` is only provided in bionic for some reason, so we have
# to acquire the deb from a mirror.
wget http://kr.archive.ubuntu.com/ubuntu/pool/main/d/dict-moby-thesaurus/dict-moby-thesaurus_1.0-6.4_all.deb && sudo dpkg -i dict-moby-thesaurus_1.0-6.4_all.deb


#█████████████████████████████████████████████████████████████████████████████#
#                                    FONTS                                    #
#█████████████████████████████████████████████████████████████████████████████#


# For downloading google fonts:
sudo apt install -y typecatcher
# Fonts acquired through typecatcher:
## (MOSTLY DISPLAY FONTS)
# Anton
# B612 Mono
# Bungee
# Fira Sans
# Fredoka One
# Lora
# Mukta
# Nanum Pen Script
# Noto Sans JP
# Nunito
# Secular One
# Permanent Marker
# Pirata One
# Playfair Display
# Press Start 2P
# Red Hat Mono
# Rubik
# Rubik Mono One
# Questrial
# VT323
# Zen Maru Gothic

sudo apt install -y \
"fonts-aoyagi*" \
fonts-bebas-neue \
fonts-cabin \
fonts-cantarell \
fonts-comfortaa \
fonts-ebgaramond \
fonts-firacode \
fonts-font-awesome \
fonts-hack \
fonts-ibm-plex \
fonts-inter \
fonts-lato \
fonts-mathjax \
fonts-mathjax-extras \
fonts-mikachan \
fonts-misaki \
fonts-moe-standard-kai \
fonts-mononoki \
fonts-motoya-l-cedar \
fonts-motoya-l-maruberi \
fonts-mplus \
fonts-noto-cjk \
fonts-noto-color-emoji \
fonts-oxygen \
fonts-roboto \
fonts-roboto-slab \
"fonts-sawarabi*" \
fonts-seto \
fonts-ubuntu \
fonts-ubuntu-console \
fonts-umeplus \
fonts-vollkorn \
;


#= EMOJI (unicode)
# It can take awhile for unicode updates to hit whatever dist ver you're on.
# For example, the "LATEST" package for Focal is `fonts-noto-color-emoji (0~20200916-1~ubuntu20.04.1)`
# Which supports...... Unicode 13.....
# Emoji on buntu are provided through the "Google Noto Color Emoji" fonts pack:
#   `fonts-noto-color-emoji`
# You can grab the deb package for the latest ubuntu release from:
# https://launchpad.net/ubuntu/+source/fonts-noto-color-emoji
# eg, from Kinetic (which is unicode 15):
wget https://launchpad.net/ubuntu/+archive/primary/+files/fonts-noto-color-emoji_2.038-1_all.deb

# Rebuild font database
sudo fc-cache -f -v


#=== FONT manager gui
sudo add-apt-repository -y ppa:font-manager/staging
sudo apt update
sudo apt install -y font-manager

#-----------------------------------------------------------------------------#
#                                Node & NPM: n                                #
#-----------------------------------------------------------------------------#


# n (node manager) npm
# ====================
curl -L https://git.io/n-install | N_PREFIX=~/.n bash -s -- -y

# npm
# ===
# Update/install npm:
npm i -g npm

#=== npm packages
# pprint web pages:
npm i -g percollate

# vscode dev stuff:
npm i -g yo generator-code vsce

# Handy git tool:
npm i -g git-branch-select
# then alias it:
git config --global alias.bs branch-select

# web-app --> 'desktop' app:
npm i -g nativefier


#-----------------------------------------------------------------------------#
#                                Stuff from GH                                #
#-----------------------------------------------------------------------------#


# Externally Sourced (GH)
# =======================
# TODO(evan): would be hella dope to have a script to get latest vers of these apps.
#             There's no damn way there isn't a gist out there to cut the release
#             ver (AND IDEALLY the linux pkg name for amd64--wildcard match and prioritize
#             .deb > .AppImage > .tar > etc....).

#=== grv (git repo viewer)
wget -O grv https://github.com/rgburke/grv/releases/download/v0.3.2/grv_v0.3.2_linux64
chmod +x grv
# (then symlink it to local bin)

#=== FreeTube
# CHECK: https://github.com/FreeTubeApp/FreeTube/releases
wget https://github.com/FreeTubeApp/FreeTube/releases/download/v0.17.1-beta/freetube_0.17.1_amd64.deb -O freetube.deb && sudo dpkg -i freetube.deb

#=== Joplin
# Execute the blessed master script.
wget -O - https://raw.githubusercontent.com/laurent22/joplin/dev/Joplin_install_and_update.sh | bash

# Install Joplin CLI:
NPM_CONFIG_PREFIX=~/.joplin-bin npm i -g joplin; sudo ln -s ~/.joplin-bin/bin/joplin /usr/bin/joplin;

#=== Marktext
# CHECK: https://github.com/marktext/marktext/releases
wget https://github.com/marktext/marktext/releases/download/v0.17.1/marktext-amd64.deb && sudo dpkg -i marktext-amd64.deb

#=== logseq
# CHECK: https://github.com/logseq/logseq/releases
#mkdir -p ~/.Apps/AppImages
#wget https://github.com/logseq/logseq/releases/download/0.8.0/Logseq-linux-x64-0.8.0.AppImage
## just let zsh autocomplete the bullshit for you
#mv Logseq*.AppImage ~/.Apps/AppImages

# CORRECTION, I made an install script for logseq because they didn't provide one.
#   Simply: ./joplin_intall_update
# Done...

#=== AppFlowy
# https://github.com/AppFlowy-IO/AppFlowy/releases


# Misc
# ====

# Stellarium
wget https://github.com/Stellarium/stellarium/releases/download/v1.1/Stellarium-1.1.1-x86_64.AppImage


#=============================================================================#
#                                   Flatpak                                   #
#=============================================================================#

# Consider adding PPA to get the latest flatpak:
sudo add-apt-repository ppa:flatpak/stable

# Install flatpak.
sudo apt install -y flatpak

# Add the Flathub repo:
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo

# Software Flatpak plugin (OPTIONAL)
# Allows you to install apps without the CLI
# sudo apt install gnome-software-plugin-flatpak

# Apps
# ====

# Emoji-picker
flatpak install flathub it.mijorus.smile
# create keyboard shortcut for: flatpak run it.mijorus.smile

#=============================================================================#
#                       __  __              _   _                             #
#                      |  \/  |            | | (_)                            #
#                      | \  / |   ___    __| |  _    __ _                     #
#                      | |\/| |  / _ \  / _` | | |  / _` |                    #
#                      | |  | | |  __/ | (_| | | | | (_| |                    #
#                      |_|  |_|  \___|  \__,_| |_|  \__,_|                    #
#                                                                             #
#=============================================================================#

# flameshot config:
# %Y-%m-%d_%H%M%S_capture


#=============================================================================#
#                                  Dev Stuff                                  #
#=============================================================================#

# Cheat
# =====
# (CHECK LATEST VER BEFORE EXEC!)
#cd ~/.cache \
#  && wget https://github.com/cheat/cheat/releases/download/4.4.0/cheat-linux-amd64.gz \
#  && gunzip cheat-linux-amd64.gz \
#  && chmod +x cheat-linux-amd64 \
#  && mv cheat-linux-amd64 $HOME/.local/bin/cheat

mkdir -p ~/.cache && cd ~/.cache \
curl https://api.github.com/repos/cheat/cheat/releases/latest | \
jq '.assets[] | select(.name=="cheat-linux-amd64.gz") | .browser_download_url' | \
xargs wget && \
gunzip cheat-linux-amd64.gz && \
chmod +x cheat-linux-amd64 && \
mkdir -p $HOME/.local/bin && \
mv cheat-linux-amd64 $HOME/.local/bin/cheat

# On first exec, cheat will install all the extra deps:
# $ cheat
# A config file was not found. Would you like to create one now? [Y/n]:
# Would you like to download the community cheatsheets? [Y/n]:
# Cloning community cheatsheets to /home/evan/.config/cheat/cheatsheets/community.
# Enumerating objects: 329, done.
# Counting objects: 100% (329/329), done.
# Compressing objects: 100% (307/307), done.
# Total 329 (delta 33), reused 271 (delta 20), pack-reused 0
# Cloning personal cheatsheets to /home/evan/.config/cheat/cheatsheets/personal.
# Created config file: /home/evan/.config/cheat/conf.yml
# Please read this file for advanced configuration information.

# AFTER INITIAL EXEC (and cheat installs all remaining deps), symlink your conf.
#--- (before symlinking, just double check the default conf.yml paths and such)
ln -sf $HOME/.Dots/cheat_conf.yml $HOME/.config/cheat/conf.yml


# Git
# ===

#=== git-standup
# > "Recall what you did on the last working day ..or be nosy and find what someone else did."
# via curl:
curl -L https://raw.githubusercontent.com/kamranahmedse/git-standup/master/installer.sh | sudo sh

# via npm (what I opt for):
npm i -g git-standup

#=== git-hours
# > "Estimate time spent on a git repository."
# CANNOT GET IT TO INSTALL; DEPS FUBAR!
# https://github.com/kimmobrunfeldt/git-hours
# Install dependency first:
#npm i -g nodegit

# Install package (troublesome--actually was unable to install, dgaf):
#npm i -g git-hours

#=== git-dev-time
# > "Calculate an estimate of the time a user spend on a repository."
# https://www.npmjs.com/package/git-dev-time
# https://github.com/vantezzen/git-dev-time#readme
npm i -g git-dev-time


#=============================================================================#
#                                   Browser                                   #
#=============================================================================#


#          ☠️☠️☠️ CHROMIUM-based BROWSERS FREEZING ON 22.04 ☠️☠️☠️           #

#  CHROMIUM-based BROWSERS FREEZING ON 22.04
#   See: https://askubuntu.com/a/1406495

# Firefox
# =======
# First, get rid of snap shit (should have done this earlier, right after fresh install).
sudo apt remove --purge snapd

# Add the firefox ppa.
sudo add-apt-repository ppa:mozillateam/ppa

# Make sure the `firefox` package prioritizes the deb.
echo '
Package: *
Pin: release o=LP-PPA-mozillateam
Pin-Priority: 1001
' | sudo tee /etc/apt/preferences.d/mozilla-firefox

# Update and install.
sudo apt update
sudo apt install -y firefox-esr
# NB: firefox-esr updates are slow as f sometimes (eg ~60kb/s), no idea why.


# Ungoogled chromium
# ==================
# Ref: https://github.com/ungoogled-software/ungoogled-chromium-debian#getting-obs-packages
# Add apt source.
echo 'deb http://download.opensuse.org/repositories/home:/ungoogled_chromium/Debian_Bullseye/ /' | sudo tee /etc/apt/sources.list.d/home-ungoogled_chromium.list > /dev/null

# Add key.
curl -s 'https://download.opensuse.org/repositories/home:/ungoogled_chromium/Debian_Bullseye/Release.key' | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/home-ungoogled_chromium.gpg > /dev/null

# Install.
sudo apt update && sudo apt install -y ungoogled-chromium


# Brave
# =====
# Deps.
sudo apt install -y apt-transport-https curl

# Brave key.
sudo curl -fsSLo /usr/share/keyrings/brave-browser-archive-keyring.gpg https://brave-browser-apt-release.s3.brave.com/brave-browser-archive-keyring.gpg

# Deb apt source.
echo "deb [signed-by=/usr/share/keyrings/brave-browser-archive-keyring.gpg arch=amd64] https://brave-browser-apt-release.s3.brave.com/ stable main" | sudo tee /etc/apt/sources.list.d/brave-browser-release.list

# Install.
sudo apt update && sudo apt install -y brave-browser





#=============================================================================#
#                                     Dev                                     #
#=============================================================================#

# ============ #
# TEXT EDITORS #
# ============ #

# Sublime
# =======
# Get the key.
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/sublimehq-archive.gpg

# Add apt source.
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list

# Install.
sudo apt update && sudo apt install -y sublime-text

# SETTINGS
#{
#    "always_show_minimap_viewport": true,
#    "auto_find_in_selection": true,
#    "bold_folder_labels": true,
#    "caret_extra_width": 2,
#    "caret_style": "smooth",
#    "detect_indentation": false,
#    "draw_minimap_border": true,
#    "ensure_newline_at_eof_on_save": true,
#    "fold_buttons": false,
#    "font_face": "Hack",
#    "font_size": 12,
#    //"highlight_line": true,
#    "hot_exit": false,
#    "ignored_packages":
#    [
#        "Vintage"
#    ],
#    "index_files": false,
#    "margin": 2,
#    "preview_on_click": false,
#    "remember_workspace": false,
#    "rulers":
#    [
#        79
#    ],
#    "tab_size": 4,
#    "translate_tabs_to_spaces": true,
#    "trim_trailing_white_space_on_save": true,
#    "auto_complete": false,
#    "color_scheme": "Packages/Colorsublime - Themes/Piodine.tmTheme",
#}

# Syntax specific: markdown
# -------------------------
#{
#    "color_scheme": "Packages/Colorsublime - Themes/Bijan.tmTheme",
#}

#==== KEYBINDINGS
# in Preferences/Key Bindings:
#[
#    {
#    "keys": ["alt+shift+c"], "command": "toggle_color_scheme",
#    "args":
#        {
#            "color_scheme_1": "Packages/Colorsublime - Themes/cache/Colorsublime-Themes-master/themes/Piodine.tmTheme",
#            "color_scheme_2": "Packages/Colorsublime - Themes/cache/Colorsublime-Themes-master/themes/Github-v2.tmTheme",
#            "color_scheme_3": "Packages/Colorsublime - Themes/cache/Colorsublime-Themes-master/themes/Spring.tmTheme",
#        }
#    },
#    { "keys": ["ctrl+k", "ctrl+q"], "command": "toggle_setting", "args": {"setting": "word_wrap"} },
#    { "keys": ["f7"], "command": "noop" },
#    { "keys": ["ctrl+b"], "command": "noop" },
#    { "keys": ["ctrl+shift+b"], "command": "noop", "args": {"select": true} },
#    { "keys": ["ctrl+break"], "command": "noop" },
#]

# VSCode
# ======
# whatever, just download it and sync, its deb does everything for you.

#=============================================================================#
#                                                                             #
#                          ██████   ██████  ████████                          #
#                          ██   ██ ██    ██    ██                             #
#                          ██   ██ ██    ██    ██                             #
#                          ██   ██ ██    ██    ██                             #
#                          ██████   ██████     ██                             #
#                                                                             #
#                                                                             #
#                           ██████ ███████  ██████                            #
#                          ██      ██      ██                                 #
#                          ██      █████   ██   ███                           #
#                          ██      ██      ██    ██                           #
#                           ██████ ██       ██████                            #
#                                                                             #
#=============================================================================#

#=============================================================================#
#                                   Editors                                   #
#=============================================================================#

# VIM
# ===
# Ehhhhhh.....................................................
# Need to proper tinker with vimrc, but here are some other options:

#---- Spacevim
#curl -sLf https://spacevim.org/install.sh | bash



#=======================================================================================#
#      ad88888ba                                                                        #
#     d8"     "8b                            ,d                                         #
#     Y8,                                    88                                         #
#     `Y8aaaaa,    8b       d8  ,adPPYba,  MM88MMM   ,adPPYba,  88,dPYba,,adPYba,       #
#       `"""""8b,  `8b     d8'  I8[    ""    88     a8P_____88  88P'   "88"    "8a      #
#             `8b   `8b   d8'    `"Y8ba,     88     8PP"""""""  88      88      88      #
#     Y8a     a8P    `8b,d8'    aa    ]8I    88,    "8b,   ,aa  88      88      88      #
#      "Y88888P"       Y88'     `"YbbdP"'    "Y888   `"Ybbd8"'  88      88      88      #
#                      d8'                                                              #
#                     d8'                                                               #
#                                                                                       #
#   ad88888ba                                 88                                        #
#  d8"     "8b                ,d       ,d     ""                                        #
#  Y8,                        88       88                                               #
#  `Y8aaaaa,     ,adPPYba,  MM88MMM  MM88MMM  88  8b,dPPYba,    ,adPPYb,d8  ,adPPYba,   #
#    `"""""8b,  a8P_____88    88       88     88  88P'   `"8a  a8"    `Y88  I8[    ""   #
#          `8b  8PP"""""""    88       88     88  88       88  8b       88   `"Y8ba,    #
#  Y8a     a8P  "8b,   ,aa    88,      88,    88  88       88  "8a,   ,d88  aa    ]8I   #
#   "Y88888P"    `"Ybbd8"'    "Y888    "Y888  88  88       88   `"YbbdP"Y8  `"YbbdP"'   #
#                                                               aa,    ,88              #
#                                                                "Y8bbdP"               #
#                                                                                       #
#=======================================================================================#


# Keyboard
# ========
#---- misc/app
flameshot gui    # Print

#---- Disable insert
# Figure out what is mapped to insert key
xmodmap -pke | grep -i insert

# Map ins key to null in ~/.Xmodmap
echo "keycode 90 =" >> ~/.Xmodmap


# Thunar custom actions
# =====================
#---- Replace spaces, format name to all lower
for file in %N; do mv "$file" "$(echo "$file" | tr -s ' ' | tr ' A-Z' '-a-z' | tr -s '-' | tr -c '[:alnum:][:cntrl:].' '-')"; done

#---- Open directory in sublime
/usr/bin/subl .

#---- Copies full pathname for a file or folder  to clipboard
echo -n %f | xclip -selection "clipboard"

#---- Touch .txt
/usr/bin/touch notes.txt

#---- Open guake here
guake -n '%d%f ' -r "%f " --show -e "cd '%f' && ls -1FSshX --file-type"



# Keychron K2 keyboard FN keys fix
# ================================
sudo vi /etc/modprobe.d/hid_apple.conf
# add the following:
options hid_apple fnmode=2

# save, close, update & reboot
sudo update-initramfs -u && reboot

#=============================================================================#
#                                                                             #
#                ██████  ███    ██  ██████  ███    ███ ███████                #
#               ██       ████   ██ ██    ██ ████  ████ ██                     #
#               ██   ███ ██ ██  ██ ██    ██ ██ ████ ██ █████                  #
#               ██    ██ ██  ██ ██ ██    ██ ██  ██  ██ ██                     #
#                ██████  ██   ████  ██████  ██      ██ ███████                #
#                                                                             #
#=============================================================================#

# Keyboard shortcuts
# ==================

#------  Remapping super# (eg super1, super2 ...)
# FROM: https://unix.stackexchange.com/questions/510375/super1-super2-super3-etc-keys-can-not-be-remapped-in-gnome
#
# You need to first clear the existing super<NUM> mappings.
# To do this, open dconf-editor:
#   - Find the keybindings that are using super<NUM>
#     - These are typically `org/gnome/shell/keybindings/switch-to-application-NUM`
#   - Uncheck "Use default value"
#   - Clear it by entering "Custom value" empty keybinding: `[]`
# You should now be able to properly bind those keys in keyboard shortcuts!

# SETTING KEYBINDINGS THROUGH CLI `dconf`:
## ACTUALLY, IT'S NOT AS EASY TO USE AS `gsettings` (esp. unsetting), LET'S USE THAT INSTEAD!
#dconf write /org/gnome/desktop/wm/keybindings/switch-to-workspace-1 "['<Super>1']" && \
#dconf write /org/gnome/desktop/wm/keybindings/switch-to-workspace-2 "['<Super>2']" && \
#dconf write /org/gnome/desktop/wm/keybindings/switch-to-workspace-3 "['<Super>3']" && \
#dconf write /org/gnome/desktop/wm/keybindings/switch-to-workspace-4 "['<Super>4']" && \
#dconf write /org/gnome/desktop/wm/keybindings/switch-to-workspace-5 "['<Super>5']";

#dconf write /org/gnome/desktop/wm/keybindings/move-to-workspace-1 "['<Primary><Alt>1']" && \
#dconf write /org/gnome/desktop/wm/keybindings/move-to-workspace-2 "['<Primary><Alt>2']" && \
#dconf write /org/gnome/desktop/wm/keybindings/move-to-workspace-3 "['<Primary><Alt>3']" && \
#dconf write /org/gnome/desktop/wm/keybindings/move-to-workspace-4 "['<Primary><Alt>4']" && \
#dconf write /org/gnome/desktop/wm/keybindings/move-to-workspace-5 "['<Primary><Alt>5']";

#=== UNSET CURRENT Super<NUM> keybindings:
gsettings set org.gnome.shell.keybindings switch-to-application-1 "[]" && \
gsettings set org.gnome.shell.keybindings switch-to-application-2 "[]" && \
gsettings set org.gnome.shell.keybindings switch-to-application-3 "[]" && \
gsettings set org.gnome.shell.keybindings switch-to-application-4 "[]" && \
gsettings set org.gnome.shell.keybindings switch-to-application-5 "[]" && \
gsettings set org.gnome.shell.keybindings switch-to-application-6 "[]" && \
gsettings set org.gnome.shell.keybindings switch-to-application-7 "[]" && \
gsettings set org.gnome.shell.keybindings switch-to-application-8 "[]" && \
gsettings set org.gnome.shell.keybindings switch-to-application-9 "[]";

#=== SET WORKSPACE KEYBINDINGS:
# switch
gsettings set org.gnome.desktop.wm.keybindings switch-to-workspace-1 "['<Super>1']" && \
gsettings set org.gnome.desktop.wm.keybindings switch-to-workspace-2 "['<Super>2']" && \
gsettings set org.gnome.desktop.wm.keybindings switch-to-workspace-3 "['<Super>3']" && \
gsettings set org.gnome.desktop.wm.keybindings switch-to-workspace-4 "['<Super>4']" && \
gsettings set org.gnome.desktop.wm.keybindings switch-to-workspace-5 "['<Super>5']";
# move
gsettings set org.gnome.desktop.wm.keybindings move-to-workspace-1 "['<Primary><Alt>1']" && \
gsettings set org.gnome.desktop.wm.keybindings move-to-workspace-2 "['<Primary><Alt>2']" && \
gsettings set org.gnome.desktop.wm.keybindings move-to-workspace-3 "['<Primary><Alt>3']" && \
gsettings set org.gnome.desktop.wm.keybindings move-to-workspace-4 "['<Primary><Alt>4']" && \
gsettings set org.gnome.desktop.wm.keybindings move-to-workspace-5 "['<Primary><Alt>5']";


# Bluetooth (or other settings in gnome gui):
gnome-control-center bluetooth


# GNOME LOCK SCREEN LID
# =====================
# Nothing has worked so far with the T14.
# I tried to edit the relevant values in /etc/systemd/logind.conf
# DID NOT WORK!
# Nor does `sudo systemctl lock`
# There seems to be some mixed terminology here in the documentation and community and settings:
## 'SUSPEND' means lock?
##  what does hibernate mean??

# WHAT WORKED??? Gnome tweaks: suspend when lid closed




#=============================================================================#
#                                                                             #
#                  ███████ ██   ██ ████████ ██████   █████                    #
#                  ██       ██ ██     ██    ██   ██ ██   ██                   #
#                  █████     ███      ██    ██████  ███████                   #
#                  ██       ██ ██     ██    ██   ██ ██   ██                   #
#                  ███████ ██   ██    ██    ██   ██ ██   ██                   #
#                                                                             #
#=============================================================================#



#=============================================================================#
#               Stuff That Still Kinda Sucks About Installation               #
#=============================================================================#

#=== GNOME
# OVERALL:
#   * NO F'N settings sync
#   * absolutely requires gnome-tweaks, shell-extensions etc...
#   * do people actually use the dock who aren't macos normies
#   * panel layout, extensions, etc.. is LEAST CONFIGURABLE of ALL DEs I've used
#   * nautilus: where do I start besides "absolutely unconfigurable"...
# - """"""""""Language Support"""""""""" for Japanese (mozc) my god just kill me now
#   - 6+ times now, installs on focal and bionic, and I still don't understand why sometimes it works as expected
# - keyboard shortcuts
#   - have to gconf/dconf whatever bs to get them lol wow f that

#=== APPS
# GITHUB:
#   - wget'ing from versioned releases urls
#     - you can probaly script something ez that can figure out the version crap,
#       but you'd need to figure out how to get correct x86_64|amd64|bartleskeet.deb|AppImage|tar.gz
#       so you either case-by-case for each main target, or you waste moments of life
# App config:
#   - a LOT of apps, I have to configure manually to my preferred settings thru a gui

#=== MISC
# NEXTCLOUD
# - Do I really, truly have to add hosts via gui every damn time
# - anything executable that's sync'd gets chmod'd I guess, git whine, ughh


#=============================================================================#
#                                                                             #
#                         db                                                  #
#                        d88b                                                 #
#                       d8'`8b                                                #
#                      d8'  `8b       ,adPPYba,  ,adPPYba,                    #
#                     d8YaaaaY8b     a8P_____88  I8[    ""                    #
#                    d8""""""""8b    8PP"""""""   `"Y8ba,                     #
#                   d8'        `8b   "8b,   ,aa  aa    ]8I                    #
#                  d8'          `8b   `"Ybbd8"'  `"YbbdP"'                    #
#                                                                             #
#=============================================================================#
# who actually has time for this

# Gnome/GTK themes
# ================
# TODO
# Browsing: https://www.gnome-look.org/s/Gnome/browse/

https://github.com/vinceliuice/Orchis-theme

###############################################################################


# CREDITS
# =======
# - jellyfish text art: https://textart.sh/topic/jellyfish
