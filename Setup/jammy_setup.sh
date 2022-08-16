#=============================================================================#
#                                                                             #
#                    ██  █████  ███    ███ ███    ███ ██    ██                #
#                    ██ ██   ██ ████  ████ ████  ████  ██  ██                 #
#                    ██ ███████ ██ ████ ██ ██ ████ ██   ████                  #
#               ██   ██ ██   ██ ██  ██  ██ ██  ██  ██    ██                   #
#                █████  ██   ██ ██      ██ ██      ██    ██                   #
#                                                                             #
#=============================================================================#

#=============================================================================#
#                            ______                                           #
#                           |  ____|                                          #
#                           | |__     _ __   __   __                          #
#                           |  __|   | '_ \  \ \ / /                          #
#                           | |____  | | | |  \ V /                           #
#                           |______| |_| |_|   \_/                            #
#                                                                             #
#=============================================================================#

# Remove bloat
# ============
sudo apt remove --purge firefox snapd
sudo apt autoremove

# Gnome stuff
# ===========
sudo apt install -y gnome-tweaks

#-----------------------------------------------------------------------------#
#                           ___   _   _   ___      _                          #
#                          / __| | | | | |   \    /_\                         #
#                         | (__  | |_| | | |) |  / _ \                        #
#                          \___|  \___/  |___/  /_/ \_\                       #
#                                                                             #
#-----------------------------------------------------------------------------#
# (USING VERSIONS AT THE TIME OF INSTALLATION)

# Get the network installer deb.
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb

# (Remove any existing cuda keys that may have been installed by default buntu 3rd party setup):
sudo apt-key del 7fa2af80

# Install CUDA keys.
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Install CUDA.
sudo apt update && sudo apt install -y cuda

# CUDA SDK packages (after reboot!).
sudo apt install -y libcudnn8

#=============================================================================#
#                                     Deps                                    #
#=============================================================================#

sudo apt install -y curl fuse git git-extras git-crypt ffmpeg python3-pip ssh xclip vim zsh

# Shell.


# Python.
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

#=============================================================================#
#                                 Environment                                 #
#=============================================================================#
# These steps all assume you have installed the deps above.

# PATH
# ====
# Default: 
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"

# Changed:
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/home/evan/.local/bin"

# Oh-my-zsh
# =========
sudo apt install -y git zsh && sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

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

# Add the following path stuff to the shell config file.
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

#=============================================================================#
#                               Settings/Config                               #
#=============================================================================#

# SSH key
# =======
sudo apt install -y ssh && ssh-keygen -t ed25519 -C 'evdcush@protonmail.com'
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519
clip ~/.ssh/id_ed25519.pub


# Japanese (WIP, STILL NOT WORKING AS EXPECTED)
# ========
# Open "Region & Language" in Settings.
# It should prompt to "properly" install some packages, namely:
ibus-mozc
mozc-utils-gui
hunspell-en-au
language-pack-gnome-ja
hunspell-en-ca
hunspell-en-za
hunspell-en-gb
language-pack-ja
fonts-noto-cjk-extra
gnome-user-docs-ja




#=============================================================================#
#                                     Misc                                    #
#=============================================================================#


#=============================================================================#
#                                                                             #
#                           /\                                                #
#                          /  \     _ __    _ __    ___                       #
#                         / /\ \   | '_ \  | '_ \  / __|                      #
#                        / ____ \  | |_) | | |_) | \__ \                      #
#                       /_/    \_\ | .__/  | .__/  |___/                      #
#                                  | |     | |                                #
#                                  |_|     |_|                                #
#                                                                             #
#=============================================================================#

# General
# =======
sudo apt install -y copyq flameshot guake jq keepassxc nextcloud-desktop screenfetch

# (Patch guake if `Could not parse file "/usr/share/applications/guake.desktop": No such file or directory`)
sudo ln -sf /usr/share/guake/autostart-guake.desktop /usr/share/applications/guake.desktop

# Media.
sudo apt install -y catimg mpv sox vlc

# PDF stuff.
sudo apt install -y texlive-xetex texlive-extra-utils

# Dict.
sudo apt install -y dict dictd dict-gcide dict-wn
# `dict-moby-thesaurus` is only provided in bionic for some reason, so we have
# to acquire the deb from a mirror.
wget http://kr.archive.ubuntu.com/ubuntu/pool/main/d/dict-moby-thesaurus/dict-moby-thesaurus_1.0-6.4_all.deb && sudo dpkg -i dict-moby-thesaurus_1.0-6.4_all.deb

# Externally Sourced
# ==================
# TODO(evan): would be nice to have a script to get latest vers of these apps!

#=== Marktext
# See: https://github.com/marktext/marktext/releases
wget https://github.com/marktext/marktext/releases/download/v0.17.1/marktext-amd64.deb

#=== logseq
# See: https://github.com/logseq/logseq/releases
wget https://github.com/logseq/logseq/releases/download/0.8.0/Logseq-linux-x64-0.8.0.AppImage


# Fonts
# =====
sudo apt install -y fonts-hack fonts-inter \ 
fonts-cabin fonts-cantarell fonts-comfortaa fonts-ebgaramond \ 
fonts-firacode fonts-font-awesome fonts-mathjax fonts-mathjax-extras fonts-mikachan \ 
fonts-misaki fonts-moe-standard-kai fonts-mononoki fonts-motoya-l-cedar \ 
fonts-motoya-l-maruberi fonts-mplus fonts-noto-cjk fonts-noto-color-emoji fonts-oxygen \ 
fonts-roboto fonts-roboto-slab "fonts-sawarabi*" fonts-seto fonts-lato fonts-umeplus \ 
fonts-ubuntu fonts-ubuntu-console fonts-vollkorn


#=============================================================================#
#                                   Browser                                   #
#=============================================================================#

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


# Firefox
# =======
# First, get rid of snap shit (should have done this earlier).
sudo apt remove --purge snapd

# Install the firefox ppa.
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
#    "font_face": "Hack",
#    "font_size": 12,
#    //"highlight_line": true,
#    "hot_exit": false,
#    "ignored_packages":
#    [
#        "Vintage"
#    ],
#    "index_files": false,
#    "preview_on_click": false,
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

# VSCode
# ======




#=============================================================================#
#                                                                             #
#                  ███████ ██   ██ ████████ ██████   █████                    #
#                  ██       ██ ██     ██    ██   ██ ██   ██                   #
#                  █████     ███      ██    ██████  ███████                   #
#                  ██       ██ ██     ██    ██   ██ ██   ██                   #
#                  ███████ ██   ██    ██    ██   ██ ██   ██                   #
#                                                                             #
#=============================================================================#


# Keychron keyboard fn keys
# =========================
# Create/edit file to have this line.
sudo echo 'options hid_apple fnmode=2' >> /etc/modprobe.d/hid_apple.conf

# Run commands:
sudo update-initramfs -u
reboot





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


# Gnome/GTK themes
# ================
# TODO
# Browsing: https://www.gnome-look.org/s/Gnome/browse/

https://github.com/vinceliuice/Orchis-theme
