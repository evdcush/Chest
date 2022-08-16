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

sudo apt install -y curl fuse git git-extras git-crypt ffmpeg imagemagick python3-pip ssh xclip vim zsh

# Real shit.
sudo apt install libboost-all-dev mpich swig
sudo apt install gcc-12 g++-12
# Can optionally setup alternatives:
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# Extras.
sudo apt install -y flac lame x264 x265


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
# (It's worth starting a new sess after install)
# Install zsh plugins:
git clone --depth=1 https://github.com/jocelynmallon/zshmarks.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zshmarks
git clone --depth=1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# ==== #
# 𝗗𝗢𝗧𝗦 #
# ==== #
ln -sf $HOME/Chest/Dots $HOME/.Dots

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
sudo apt install -y copyq deluge flameshot gparted guake inkscape keepassxc nextcloud-desktop nomacs pandoc redshift wkhtmltopdf

# (Patch guake if `Could not parse file "/usr/share/applications/guake.desktop": No such file or directory`)
sudo ln -sf /usr/share/guake/autostart-guake.desktop /usr/share/applications/guake.desktop

#== CLI.
sudo apt install -y aria2 bat delta duf htop jq screenfetch tree

#== Media.
sudo apt install -y catimg mpv sox vlc

#== Archive.
sudo apt install -y p7zip-full unar

#== PDF stuff.
sudo apt install -y texlive-xetex texlive-extra-utils

#== Dict.
sudo apt install -y dict dictd dict-gcide dict-wn
# `dict-moby-thesaurus` is only provided in bionic for some reason, so we have
# to acquire the deb from a mirror.
wget http://kr.archive.ubuntu.com/ubuntu/pool/main/d/dict-moby-thesaurus/dict-moby-thesaurus_1.0-6.4_all.deb && sudo dpkg -i dict-moby-thesaurus_1.0-6.4_all.deb

# Externally Sourced (GH)
# =======================
# TODO(evan): would be hella dope to have a script to get latest vers of these apps.
#             There's no damn way there isn't a gist out there to cut the release
#             ver (AND IDEALLY the linux pkg name for amd64--wildcard match and prioritize
#             .deb > .AppImage > .tar > etc....).

#=== FreeTube
# CHECK: https://github.com/FreeTubeApp/FreeTube/releases
wget https://github.com/FreeTubeApp/FreeTube/releases/download/v0.17.1-beta/freetube_0.17.1_amd64.deb -O freetube.deb && sudo dpkg -i freetube.deb

#=== Joplin
# Execute the blessed master script.
wget -O - https://raw.githubusercontent.com/laurent22/joplin/dev/Joplin_install_and_update.sh | bash

#=== Marktext
# CHECK: https://github.com/marktext/marktext/releases
wget https://github.com/marktext/marktext/releases/download/v0.17.1/marktext-amd64.deb && sudo dpkg -i marktext-amd64.deb

#=== logseq
# CHECK: https://github.com/logseq/logseq/releases
mkdir -p ~/.Apps/AppImages
wget https://github.com/logseq/logseq/releases/download/0.8.0/Logseq-linux-x64-0.8.0.AppImage
# just let zsh autocomplete the bullshit for you
mv Logseq*.AppImage ~/.Apps/AppImages



#-----------------------------------------------------------------------------#
#                    Sourced through other package managers                   #
#-----------------------------------------------------------------------------#

# n (node manager) npm
# ====================
curl -L https://git.io/n-install | N_PREFIX=~/.n bash -s -- -y

# npm
# ---
npm i -g npm && npm i -g percolalte


#=============================================================================#
#                                  Dev Stuff                                  #
#=============================================================================#

# Cheat
# =====
# (CHECK LATEST VER BEFORE EXEC!)
cd /tmp \
  && wget https://github.com/cheat/cheat/releases/download/4.3.1/cheat-linux-amd64.gz \
  && gunzip cheat-linux-amd64.gz \
  && chmod +x cheat-linux-amd64 \
  && sudo mv cheat-linux-amd64 /usr/local/bin/cheat \
  && mkdir -p ~/.config/cheat && cheat --init > ~/.config/cheat/conf.yml \
  && ln -sf ~/.Dots/cheat_conf.yml ~/.config/cheat/conf.yml



#=============================================================================#
#                                   Browser                                   #
#=============================================================================#

# Firefox
# =======
# First, get rid of snap shit (should have done this earlier).
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
