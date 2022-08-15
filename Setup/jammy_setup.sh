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
sudo apt remove --purge snapd
sudo apt autoremove

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

sudo apt install -y curl git vim zsh

# Shell.


# Python.


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

#=============================================================================#
#                               Settings/Config                               #
#=============================================================================#


# Japanese
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

# VSCode
# ======







sudo vi /etc/modprobe.d/hid_apple.conf