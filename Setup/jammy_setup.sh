# 2022 github.com/evdcush
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

# Get the image.
wget https://ftp.riken.jp/Linux/ubuntu-releases/jammy/ubuntu-22.04.1-desktop-amd64.iso

# Burn it to a USB using Etcher.
#   NB: don't be a dd hero here; it's either etcher or pain
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

# Gnome stuff
# ===========
sudo apt install -y gnome-tweaks

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
#                        █████  ██████  ██████  ███████                       #
#                       ██   ██ ██   ██ ██   ██ ██                            #
#                       ███████ ██████  ██████  ███████                       #
#                       ██   ██ ██      ██           ██                       #
#                       ██   ██ ██      ██      ███████                       #
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
# whatever, just download it and sync, its deb does everything for you.



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
