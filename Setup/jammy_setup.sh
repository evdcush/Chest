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
#    { "keys": ["alt+z"], "command": "toggle_setting", "args": {"setting": "word_wrap"} },
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
sudo rm /etc/apt/apt.conf.d/20apt-esm-hook.conf
# NOTE: You will need to consistently remove this hook!
#       Updates sometimes restore it!
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

# Unf*cking a CUDA installation that decides it wanted to be f*cked one day
# -------------------------------------------------------------------------

# Have you ever booted your machine, and it decides that it no longer has
# working cuda driver and your display is stuck in low-res?
#   > But I didn't even do anything! I didn't upgrade CUDA or change anything!
#   > I tried to fix/update/downgrade CUDA through APT, but APT is totally
#     borked! Holding "Broken packages"? But they're NVIDIA's damn packages!
#     It's totally stuck in a broken deps loop, and `apt install --fix-broken`
#     does nothing!
#
# My dear fren, this is the CUDA way.
# We do not reason with CUDA.
# We do not negotiate with CUDA.
# Only scorched-earth methods are effective.
# Probably good to make a backup at this point, because now we must
# coldly commit to our MAD strategy.


# Let's set the stage,
# you're video display looks off, and you see there are an issue with your
# (CUDA) video drivers.
# You attempt to upgrade your packages,
# you see some f'ing bullsh*t like:

# > Reading package lists... Done
# > Building dependency tree
# > Reading state information... Done
# > You might want to run 'apt --fix-broken install' to correct these.
# > The following packages have unmet dependencies:
# >  cuda-drivers : Depends: cuda-drivers-550 (= 550.54.15-1) but it is not installed
# >  libnvidia-decode-545 : Depends: libnvidia-compute-545 (= 545.23.08-0ubuntu1) but it is not installed
# > E: Unmet dependencies. Try 'apt --fix-broken install' with no packages (or specify a solution).

# FIRST STEP!
# ===========
# Switch your video driver provider from nvidia garb to noveau.
# This will allow you to rip and tear CUDA and nvidia in your system without
# affecting your display.

# SECOND STEP!
# ============
# Do not attempt to try and resolve the broken deps issue through apt.
# It won't work.
# It's now time to remove packages manually through `dpkg`, eg:
# Start with the generic packagers,
# the dpkg system will warn you which specific/versioned packages depend on it,
# and then you just remove them, step-by-step.
# For example:
sudo dpkg -r cuda
sudo dpkg -r cuda-drivers
sudo dpkg -r cuda-runtime
sudo dpkg -r cuda-12-3

# sudo dpkg -r cuda-drivers
# dpkg: dependency problems prevent removal of cuda-drivers:
#  cuda-runtime-12-3 depends on cuda-drivers (>= 545.23.08).
#
# dpkg: error processing package cuda-drivers (--remove):
#  dependency problems - not removing
# Errors were encountered while processing:
#  cuda-drivers
# Here's an example trace::

sudo dpkg -r cuda-runtime-12-3
dpkg: dependency problems prevent removal of cuda-runtime-12-3:
 cuda-demo-suite-12-3 depends on cuda-runtime-12-3.
 cuda-12-3 depends on cuda-runtime-12-3 (>= 12.3.2).

dpkg: error processing package cuda-runtime-12-3 (--remove):
 dependency problems - not removing
Errors were encountered while processing:
 cuda-runtime-12-3
sudo dpkg -r cuda-demo-suite-12-3
dpkg: dependency problems prevent removal of cuda-demo-suite-12-3:
 cuda-12-3 depends on cuda-demo-suite-12-3 (>= 12.3.101).

dpkg: error processing package cuda-demo-suite-12-3 (--remove):
 dependency problems - not removing
Errors were encountered while processing:
 cuda-demo-suite-12-3
sudo dpkg -r cuda-12-3
(Reading database ... 498017 files and directories currently installed.)
Removing cuda-12-3 (12.3.2-1) ...
sudo dpkg -r cuda-demo-suite-12-3
(Reading database ... 498015 files and directories currently installed.)
Removing cuda-demo-suite-12-3 (12.3.101-1) ...






#=============================================================================#
#                                     Deps                                    #
#=============================================================================#

# Generally needed.
sudo apt install -y curl git git-crypt ffmpeg imagemagick python3-pip ssh xclip vim zsh v4l-linux

# Real shit.
sudo apt install libboost-all-dev mpich swig liblapack-dev libblas-dev
sudo apt install gcc-12 g++-12
# Can optionally setup alternatives:
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12

sudo apt install -y libhdf5-dev hdf5-tools

# Graph.
sudo apt install -y graphviz graphviz-dev

# Extras.
sudo apt install -y flac git-extras lame x264 x265

# Git
# ===
#--- git-lfs
# Add the source:
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

# Install:
sudo apt update && sudo apt install -y git-lfs;

# Python (pyenv)
# ==============
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

#---- system python
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update && sudo apt upgrade

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

# The general "add zsh plugin to oh-my-zsh" snippet:
## Clone repo into zsh custom plugins:
git clone --depth=1 https://github.com/<owner>/<repo>.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/<repo>

## Add it to .zshrc plugins
#plugins=( history colored-man-pages zsh-syntax-highlighting etc... )


#--- Experimental
git clone --depth=1 \
https://github.com/LucasLarson/gunstage.git \
${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/gunstage

#--- resh
# "Shell history in context"
# eg: like fzf, but with further contextual hints/ordering
curl -fsSL https://raw.githubusercontent.com/curusarn/resh/master/scripts/rawinstall.sh | bash

###############################################################################
# Try atuin? https://github.com/atuinsh/atuin
#################################################################################


# ==== #
# 𝗗𝗢𝗧𝗦 #
# ==== #
ln -sf $HOME/Chest/Dots $HOME/.Dots


#-----------------------------------------------------------------------------#
#                                     Keys                                    #
#-----------------------------------------------------------------------------#

# GPG
# ===
#==== Generate
gpg --full-generate-key

# This will ask you for:
#   - what kind of key you want: default (RSA, RSA)
#   - bit length of new key: 4096
#   - expiry: choose whatever is appropriate
#   - Key:
#     - Real Name
#     - Email
#     - Comment

#==== See your new key:
gpg --list-secret-keys --keyid-format=long

#==== Copy your new pubkey in ASCII armor format
# Copy the long-form of the key
# (On the `sec` line, this is the 16-char hex that follows `rsa4096/<KEY>`)
# Then export
gpg --armor --export <YOUR_KEY> | xclip -sel clip



#█████████████████████████████████████████████████████████████████████████████#
#                                     GIT                                     #
#█████████████████████████████████████████████████████████████████████████████#


# Git
# ===
# You can symlink your Dots/gitconfig to ~/.gitconfig
#ln -sf $HOME/.Dots/gitconfig $HOME/.gitconfig
# You should definitely symlink your global gitignore:
ln -sf $HOME/.Dots/global_gitignore $HOME/.gitignore

# Or specify options manually:
git config --global user.name "Evan C."
git config --global user.email "evdcush@protonmail.com"
git config --global init.defaultBranch master
git config --global core.excludesFile '~/.gitignore'
git config --global help.autocorrect 8  # 0.8s to cancel an auto-corrected command.
git config --global url.'git@github.com:'.insteadOf 'https://github.com/'  # use ssh instead of https for remotes
#git config --global core.pager ''  # careful, this will also dump git log straight up

# After npm i -g git-branch-select:
git config --global alias.bs branch-select

# Figure out which signing key you want:
gpg --list-secret-keys --keyid-format=long
# Set it:
git config --local user.signingkey 0123456789ABCDEF
git config --global commit.gpgsign true


#=== TOOLING
# nbdime
## pip install -U nbdime
nbdime config-git --enable --global

# nbstripout
## pip install -U nbstripout
# Not sure, but here are the options you want:
nbstripout --install --keep-output --drop-empty-cells

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
## 𝗗𝗢𝗘𝗦 𝗡𝗢𝗧 𝗪𝗢𝗥𝗞; 𝗔𝗕𝗦𝗢𝗟𝗨𝗧𝗘 (𝗨𝗡𝗡𝗘𝗖𝗘𝗦𝗦𝗔𝗥𝗬) 𝗗𝗘𝗣 𝗛𝗘𝗟𝗟
# (ON F'N PYTHON >= 3.10, but have 3.8 installed 😐️)


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


#=============================================================================#
#                               Settings/Config                               #
#=============================================================================#

# Gnome stuff
# ===========
sudo apt install -y gnome-tweaks nautilus-extension-fma

#=== Extensions
# SYSTEM-MONITOR: https://extensions.gnome.org/extension/3010/system-monitor-next/
#                 https://github.com/mgalgs/gnome-shell-system-monitor-applet


# SSH key
# =======
sudo apt install -y ssh && \
ssh-keygen -t ed25519 -C 'evdcush@protonmail.com'
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519
xclip -sel clip ~/.ssh/id_ed25519.pub


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
gnome-shell-extension-dash-to-panel

#==== Prevent gnome from opening file-explorer in background (rather than foreground).
## REF: https://askubuntu.com/questions/1414083/gnome-file-explorer-window-opens-in-the-background-instead-of-foreground
sudo apt install -y gnome-shell-extension-no-annoyance

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

# REMOVE THE DEFAULT BOOKMARKS IN NAUTILUS
# ========================================
# https://askubuntu.com/questions/762591/how-to-remove-unwanted-default-bookmarks-in-nautilus
## Comment out the paths in both:
##   ~/.config/user-dirs.dirs
##   /etc/xdg/user-dirs.defaults



#=============================================================================#
#                                     Misc                                    #
#=============================================================================#

# hist
# ====
# atuin: https://github.com/atuinsh/atuin
# Syncs shell hist across devices.
curl --proto '=https' --tlsv1.2 -LsSf https://setup.atuin.sh | sh

# Login to your account (yknow the drill):
atuin login -u USERNAM


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
#                            NO SOUND, DUMMY OUTPUT                           #
#=============================================================================#
# A fucking nightmare, this one.
# Consulted and tried all in:
# https://askubuntu.com/questions/1406646/ubuntu-22-04-audio-output-not-working-dummy-audio
# Didn't work.

# What worked: Forcing the use of legacy HDA driver
# Something to do with SOF (Sound Open Firmware) driver, a newer intel hardware
# thing I guess.
# Discovered by `sudo dmesg | grep snd`
sudo vi /etc/default/grub

# Add line:
# GRUB_CMDLINE_LINUX_DEFAULT="quiet splash snd_hda_intel.dmic_detect=0"

# update and reboot.
sudo update-grub && reboot

# Install SOF firmware
sudo apt install -y sof-firmware
reboot


#=============================================================================#
#                          NO EXTERNAL DISPLAY (HDMI)                         #
#=============================================================================#
## PROBLEM:
#  - connect external monitor to Lenovo ThinkPad T14, Buntu 22.04 6.2.0-37-generic
#  - Unable to connect to display
## CAUSE:
#  This issue is caused by a bug in the T14 Gen3 BIOS
#  Ref: https://askubuntu.com/questions/1421990/new-lenovo-ubuntu-22-04-hdmi-output-not-detected
## FIX:
#  Use `fwupmgr` to update firmware

# Find update (unnecessary):
fwupdmgr get-devices

# Install update:
fwupdmgr get-updates

# ^ that didn't work for me, but this did:
fwupdmgr refresh && fwupdmgr update

# IF THIS DOESN'T WORK, CHANGE CONFIGURATION BIOS: from legacy to UEFI


#=============================================================================#
#                          FUCKING SNAP MOUNT FIREFOX                         #
#=============================================================================#

# You should have already uninstalled it earlier, but just for good measure:
sudo apt remove --purge snapd

# Unmount the firefox hunspell snap bullshit:
sudo umount /var/snap/firefox/common/host-hunspell

# But wait, you're not done yet you FOOL!
# It will come back when you reboot!
# Remove this turd as well:
sudo rm /etc/systemd/system/multi-user.target.wants/var-snap-firefox-common-host\\x2dhunspell.mount

# And these ones:
sudo rm /etc/systemd/system/snapd.mounts.target.wants/var-snap-firefox-common-host\\x2dhunspell.mount
sudo rm /etc/systemd/system/var-snap-firefox-common-host\\x2dhunspell.mount

# Fuck off snap ;]


#=============================================================================#
#                                                                             #
#                        █████  ██████  ██████  ███████                       #
#                       ██   ██ ██   ██ ██   ██ ██                            #
#                       ███████ ██████  ██████  ███████                       #
#                       ██   ██ ██      ██           ██                       #
#                       ██   ██ ██      ██      ███████                       #
#                                                                             #
#=============================================================================#
## GENERALLY SPEAKING
## Any app that is available through the official/master apt sources
## will be (quite a bit; exceedingly) behind the current releasre version
## that may be available through (in order of preference):
##   DEB (preferred)
##     PPA
##     GitHub (deb)
##   Flatpak
##   GitHub (AppImage or binary)
##
## So, for HEAVILY used applications and deps, always check first for their
## latest release ver, and where to find it!!

# System utils
# ============
# HDD check (for nvme)
sudo apt install -y nvme-cli

# General
# =======
sudo apt install -y fd-find locate
sudo apt install -y jq
sudo apt install -y copyq deluge gparted redshift wkhtmltopdf inxi colordiff qalc poppler-utils
sudo apt install -y units  # unit converter

# General Sourced
# ---------------
# These apps generally have very old vers in official apt repos, but can get
# elsewhere.

#=== KeePassXC
sudo add-apt-repository -y ppa:phoerious/keepassxc && \
sudo apt update && \
sudo apt install keepassxc

# NB: `keepassxc-cli` is included in the `keepassxc` package!
# You can use the CLI to retrieve totp codes and such.
# Eg:
keepassxc-cli show -t my_pwdb.kdbx myEntry

#=== Pandoc
curl https://api.github.com/repos/jgm/pandoc/releases/latest | \
jq '.assets[] | select(.name | startswith("pandoc-") and endswith("amd64.deb")) | .browser_download_url' | \
xargs wget -O software.deb && \
sudo dpkg -i software.deb \
&& rm software.deb;

#=== FLAMESHOT
# https://github.com/flameshot-org/flameshot/releases

echo "\n\nDOWNLOADING FLAMESHOT..."
echo "#########################################"
curl https://api.github.com/repos/flameshot-org/flameshot/releases/latest | \
jq '.assets[] | select(.name | startswith("flameshot-") and endswith(".ubuntu-20.04.amd64.deb")) | .browser_download_url' | \
xargs wget -O software.deb && \
sudo dpkg -i software.deb \
&& rm software.deb;
echo "\n\nFINISHED FLAMESHOT"
echo "#########################################\n"

#=== INKSCAPE
sudo add-apt-repository -y ppa:inkscape.dev/stable && \
sudo apt update && \
sudo apt install -y inkscape


#=========================================================
# Libreoffice
# ===========
#------ Core suite
## This is way too much
sudo add-apt-repository -y ppa:libreoffice/ppa
sudo apt install -y libreoffice-dev \
libreoffice-wiki-publisher \
libreoffice-texmaths \
libreoffice-templates \
libreoffice-script-provider-python \
libreoffice-calc \
libreoffice-math \
libreoffice-l10n-ja \
libreoffice-dmaths \
libreoffice-draw \
libreoffice-gnome \
libreoffice-systray \
libreoffice-pdfimport \
libreoffice-impress \
libreoffice-evolution

# Note:
#   - libreoffice-gnome: because if you don't, you get LO's shitty looking file-manager.
#   - libreoffice-script-provider-python: Dependency


# EXTENSIONS
# ----------
#-----  Code Highlighter 2
#   src: https://github.com/jmzambon/libreoffice-code-highlighter/releases
#   catalog: https://extensions.libreoffice.org/en/extensions/show/5814
#
# First, install deps (INCLUDED IN CORE-SUITE INSTALL LINE ABOVE!)
sudo apt install -y libreoffice-script-provider-python

# Then, get the latest release.
## I guess you cannot install from CLI, so add it in the extensions manager GUI in LO.
curl https://api.github.com/repos/jmzambon/libreoffice-code-highlighter/releases/latest | \
jq '.assets[] | select(.name | endswith(".oxt")) | .browser_download_url' | \
xargs wget -O codehighlighter2.oxt


## Load LibreOffice config
# YOU NEED TO COPY YOUR USER PROFILE FROM CONFIG:
# `$HOME/.config/libreoffice/4/user`

#---- Keyboard shortcuts
# libreoffice
# libreoffice /home/foo-bar/InDaCloud/WorldChangingIdeas/notez.odt

# "Rambox"
##  https://rambox.app/download-linux/
##  Combines all the different services (email, slack, discord, etc..)
##  into a single application (SEEMS HANDY!).


#---------------------------------------------------------

# ETHERPAD
# ========
# collaborative editing online editor.



#=========================================================

#==== Pandoc
## Maybe get it from GH releases instead of apt sources:
# https://github.com/jgm/pandoc/releases

#==== Nextcloud
# For Nextcloud, you may want to consider not using the default deb file available
# in the official apt repos, and instead try out the AppImage, which is what
# Nextcloud seems to "officially" endorse.
#   BUT, if you do install deb package, you probaly should add the Nextcloud PPA
#   first, so you can get the latest builds (since official repo vers are often
#   significantly older than latest):
sudo add-apt-repository -y ppa:nextcloud-devs/client && \
sudo apt update && \
sudo apt install -y nextcloud-desktop;
# (Consider `nautilus-nextcloud` as well)
## SHOULD YOU ENCOUNTER AUTH ISSUES, such as:
## "the password you use to log in to your computer no longer matches
##  that of your login keyring"
## Or other keyring issues, here's what worked for me:
## rm -v ~/.local/share/keyrings/*.keyring && reboot


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

# NB, edit line:
subl --new-window --wait %(file_path)s:%(line_number)s

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

#=== nteract
# https://github.com/nteract/nteract/releases
## eg: `nteract_0.28.0_amd64.deb`

#=== phoronix
# https://github.com/phoronix-test-suite/phoronix-test-suite
## eg: `phoronix-test-suite_10.8.4_all.deb``
curl https://api.github.com/repos/phoronix-test-suite/phoronix-test-suite/releases/latest | \
jq '.assets[] | select(.name | startswith("phoronix-test-suite_") and endswith("_all.deb")) | .browser_download_url' | \
xargs wget -O foo.deb && \
sudo dpkg -i foo.deb \
&& rm foo.deb;

#=== ElectronMail
# https://github.com/vladimiry/ElectronMail
## Unofficial protonmail desktop app
## eg:
## https://github.com/vladimiry/ElectronMail/releases/download/v5.1.3/electron-mail-5.1.3-linux-amd64.deb
curl https://api.github.com/repos/vladimiry/ElectronMail/releases/latest | \
jq '.assets[] | select(.name | endswith("linux-amd64.deb")) | .browser_download_url' | \
xargs wget -O software.deb && \
sudo dpkg -i software.deb \
&& rm software.deb;

#===  gitty
## Contextual information about your git projects, right on the command-line
# https://github.com/muesli/gitty/releases/download/v0.7.0/gitty_0.7.0_linux_amd64.deb

#=== bottom
## way better sys mon
# https://github.com/ClementTsang/bottom



#=============================================================================#


#=== Top.
sudo apt install -y htop stacer
wget https://github.com/ClementTsang/bottom/releases/download/0.6.8/bottom_0.6.8_amd64.deb && sudo dpkg -i bottom_0.6.8_amd64.deb

#== File-system
sudo apt install -y exfat-fuse exfat-utils f2fs-tools ntfs-3g zfs-fuse zfsutils-linux

#== Archive
sudo apt install -y p7zip-full unar

#== Knowledge Management
sudo apt install -y buku

#== PDF stuff.
sudo apt install -y texlive-xetex texlive-extra-utils \
texlive-bibtex-extra texlive-fonts-extra texlive-font-utils texlive-lang-cjk \
texlive-lang-japanese texlive-pstricks texlive-publishers texlive-science latexmk

## ⚠️⚠️ WARNING ⚠️⚠️
# installation attempts of `texlive*` were totally borked when I tried recently
# on jammy
# I kept getting the following bs:
#  $ sudo apt install texlive-xetex
#  Reading package lists... Done
#  Building dependency tree... Done
#  Reading state information... Done
#  Some packages could not be installed. This may mean that you have
#  requested an impossible situation or if you are using the unstable
#  distribution that some required packages have not yet been created
#  or been moved out of Incoming.
#  The following information may help to resolve the situation:
#
#  The following packages have unmet dependencies:
#   texlive-base : Depends: texlive-binaries (>= 2021.20210626)
#   texlive-latex-base : Depends: texlive-binaries (>= 2021.20210626)
#   texlive-latex-extra : Depends: texlive-binaries (>= 2021.20210626)
#                         Recommends: texlive-plain-generic but it is not installable
#   texlive-latex-recommended : Depends: texlive-binaries (>= 2021.20210626)
#   texlive-pictures : Depends: texlive-binaries (>= 2021.20210626)
#   texlive-xetex : Depends: texlive-binaries (>= 2021.20210626)
#                   Depends: tipa (>= 2:1.2-2.1) but it is not installable
#  E: Unable to correct problems, you have held broken packages.
#
####
# TOTAL FUBAR
# THE FIX:
sudo apt install -y aptitude
sudo aptitude install texlive-base

# HERE'S WHAT IT LOOKED LIKE (WTF?? 🤷🏻‍♂️)
#   sudo aptitude install texlive-base
#   The following NEW packages will be installed:
#     fonts-lmodern{a} lmodern{a} tex-common{a} texlive-base{b}
#   0 packages upgraded, 4 newly installed, 0 to remove and 0 not upgraded.
#   Need to get 35.1 MB of archives. After unpacking 87.9 MB will be used.
#   The following packages have unmet dependencies:
#    texlive-base : Depends: texlive-binaries (>= 2021.20210626) but it is not installable
#   The following actions will resolve these dependencies:
#
#        Keep the following packages at their current version:
#   1)     texlive-base [Not Installed]
#
#
#
#   Accept this solution? [Y/n/q/?] n
#   The following actions will resolve these dependencies:
#
#        Install the following packages:
#   1)     libptexenc1 [2021.20210626.59705-1build1 (jammy)]
#   2)     libteckit0 [2.5.11+ds1-1 (jammy)]
#   3)     libtexlua53 [2021.20210626.59705-1build1 (jammy)]
#   4)     libtexluajit2 [2021.20210626.59705-1build1 (jammy)]
#   5)     libzzip-0-13 [0.13.72+dfsg.1-1.1 (jammy)]
#   6)     t1utils:i386 [1.41-4build2 (jammy)]
#   7)     texlive-binaries [2021.20210626.59705-1build1 (jammy)]
#
#        Downgrade the following packages:
#   8)     libkpathsea6 [2021.20210626.59705-1ubuntu0.1 (now) -> 2021.20210626.59705-1build1 (jammy)]
#   9)     libsynctex2 [2021.20210626.59705-1ubuntu0.1 (now) -> 2021.20210626.59705-1build1 (jammy)]
#
#
#
#   Accept this solution? [Y/n/q/?]
#   The following packages will be DOWNGRADED:
#     libkpathsea6 libsynctex2
#   The following NEW packages will be installed:
#     fonts-lmodern{a} libptexenc1{a} libteckit0{a} libtexlua53{a} libtexluajit2{a} libzzip-0-13{a} lmodern{a} t1utils:i386{a} tex-common{a} texlive-base texlive-binaries{a}
#   0 packages upgraded, 11 newly installed, 2 downgraded, 0 to remove and 0 not upgraded.
#   Need to get 46.0 MB of archives. After unpacking 141 MB will be used.
#   Do you want to continue? [Y/n/?]


# Sphinx needs this to build `latexpdf`
sudo apt install -y latexmk

#== Dict.
sudo apt install -y dict dictd dict-gcide dict-wn
sudo apt install -y dict-freedict-eng-jpn dict-freedict-eng-lat
# `dict-moby-thesaurus` is only provided in bionic for some reason, so we have
# to acquire the deb from a mirror.
wget http://kr.archive.ubuntu.com/ubuntu/pool/main/d/dict-moby-thesaurus/dict-moby-thesaurus_1.0-6.4_all.deb && sudo dpkg -i dict-moby-thesaurus_1.0-6.4_all.deb


#=============================================================================#
#                                WebApp-Manager                               #
#=============================================================================#
# This tool is from Linuxmint and not available in official apt sources.
# There are 3rd-party, unofficial PPAs that provide linuxmint software
# for Ubuntu installation, such as:
#   https://launchpad.net/~kelebek333/+archive/ubuntu/mint-tools
#   ppa:kelebek333/mint-tools
# These should be considered at some point if direct wget process becomes
# tedious.
# Otherwise,
# Go here, get the latest: http://packages.linuxmint.com/pool/main/w/webapp-manager/
wget http://packages.linuxmint.com/pool/main/w/webapp-manager/webapp-manager_1.3.7_all.deb
sudo dpkg -i webapp-manager_1.3.7_all.deb
sudo apt install -f

# Adding Keyboard launch shortcuts
# ================================
# In order to launch the app directly without needing to open webapp-manager
# or navigating through your apps dashboard, find the `exec` command
# used in the webapp's `*.desktop` file. These should be located and identified
# with the following pattern:
# `$HOME/.local/share/applications/WebApp-*<app-name>*.desktop`
# eg:
cat $HOME/.local/share/applications/WebApp-ChatGPT1893.desktop
# Returns:
## [Desktop Entry]
## Version=1.0
## Name=ChatGPT
## Comment=Web App
## Exec=/var/lib/flatpak/exports/bin/io.github.ungoogled_software.ungoogled_chromium --app="https://chatgpt.com/" --class=WebApp-ChatGPT1893 --name=WebApp-ChatGPT1893 --user-data-dir=/home/evan/.local/share/ice/profiles/ChatGPT1893
## Terminal=false
## X-MultipleArgs=false
## Type=Application
## Icon=/path/to/your/selected/app/icon.svg
## Categories=GTK;WebApps;
## MimeType=text/html;text/xml;application/xhtml_xml;
## StartupWMClass=WebApp-ChatGPT1893
## StartupNotify=true
## X-WebApp-Browser=Ungoogled Chromium (Flatpak)
## X-WebApp-URL=https://chatgpt.com/
## X-WebApp-CustomParameters=
## X-WebApp-Navbar=false
## X-WebApp-PrivateWindow=false
## X-WebApp-Isolated=true
#
# Grab that exec command, and you can make a keyboard shortcut!





#=============================================================================#
#                                    Docker                                   #
#=============================================================================#
# REF: https://docs.docker.com/engine/install/ubuntu/

# Add Docker's official GPG key:
sudo apt install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install it:
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Postinstall permissions shit.
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker # (or just logout)

# Verify installation:
docker run hello-world



#█████████████████████████████████████████████████████████████████████████████#
#                                    FONTS                                    #
#█████████████████████████████████████████████████████████████████████████████#


# For downloading google fonts:
sudo apt install -y typecatcher
## If you get error:
# AttributeError: 'ElementTree' object has no attribute 'getiterator'

# Fix:
sudo sed -i 's/tree.getiterator/tree.iter/g' /usr/lib/python3/dist-packages/typecatcher_lib/Builder.py


# Fonts acquired through typecatcher:
## (MOSTLY DISPLAY FONTS)
# Anton
# B612 Mono
# Fira Sans
# Fredoka One
# Lora
# Mukta
# Nanum Pen Script
# Noto Sans JP
# Nunito
# Secular One
# Permanent Marker
# Playfair Display
# Red Hat Mono
# Rubik
# Questrial
# VT323
# Zen Maru Gothic

#===  Display
# Aboreto
# Astloch
# Bangers
# Bokor
# Bungee
# Cinzel
# Cinzel Decorative
# Forum
# Nosifer
# Pirata One
# Press Start 2P
# Rubik 80s Fade
# Rubik Mono One  #$# AND MONO?
# Silkscreen
# Special Elite
# UnifrakturMaguntia
# Tilt Prism

#=== Mono
# Spline Sans Mono
# Victor Mono

#=== Sans
# Inter Tight
# Open Sans

#=== Serif
# Adamina
# PT serif
# Lora
# Noto Serif
# Noto Serif JP
# IBM Plex Serif
# Spectral


#=== Japanese
# Yuji Hentaigana Akebono
# Rock 3D
# Tsukimi Rounded
# Monomaniac One
# Slackside One




#-----------------------------------------------------

sudo apt install -y \
"fonts-aoyagi*" \
"fonts-ebgaramond*" \
"fonts-hack*" \
"fonts-mathjax*" \
"fonts-motoya*" \
"fonts-sawarabi*" \
"fonts-takao*" \
"fonts-umeplu*" \
fonts-bebas-neue \
fonts-cabin \
fonts-cantarell \
fonts-cascadia-code \
fonts-comfortaa \
fonts-ebgaramond \
fonts-firacode \
fonts-font-awesome \
fonts-hack \
fonts-ibm-plex \
fonts-inter \
fonts-inter-variable \
fonts-jetbrains-mono \
fonts-katex \
fonts-lato \
fonts-mathjax \
fonts-mathjax-extras \
fonts-mikachan \
fonts-misaki \
fonts-moe-standard-kai \
fonts-mononoki \
fonts-motoya-l-cedar \
fonts-motoya-l-maruberi \
fonts-mph-2b-damase \
fonts-mplus \
fonts-noto-cjk \
fonts-noto-cjk-extra \
fonts-noto-color-emoji \
fonts-noto-mono \
fonts-oxygen \
fonts-roboto \
fonts-roboto-hinted \
fonts-roboto-slab \
fonts-seto \
fonts-ubuntu \
fonts-ubuntu-console \
fonts-ubuntu-title \
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
#wget https://launchpad.net/ubuntu/+archive/primary/+files/fonts-noto-color-emoji_2.038-1_all.deb
wget https://launchpad.net/ubuntu/+archive/primary/+files/fonts-noto-color-emoji_2.038-0ubuntu1_all.deb

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

# pprint markdown (pdf):
npm i -g pretty-markdown-pdf

# vscode dev stuff:
npm i -g yo generator-code vsce

# Handy git tool:
npm i -g git-branch-select
# then alias it:
git config --global alias.bs branch-select

# web-app --> 'desktop' app:
npm i -g nativefier

# Visualize stats about GH users and projects from terminal
npm i -g github-stats


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

#=== Raven Reader (RSS)
wget https://github.com/hello-efficiency-inc/raven-reader/releases/download/v1.0.78/Raven-Reader-1.0.78.AppImage


# Misc
# ====

# Stellarium
wget https://github.com/Stellarium/stellarium/releases/download/v1.1/Stellarium-1.1.1-x86_64.AppImage


# FreeTube
# ========
# CHECK: https://github.com/FreeTubeApp/FreeTube/releases
wget https://github.com/FreeTubeApp/FreeTube/releases/download/v0.17.1-beta/freetube_0.17.1_amd64.deb -O freetube.deb && sudo dpkg -i freetube.deb

## Config stuffs:
# filename pattern:
# cap_%Y-%M-%D_%H-%N-%S_%i_%s
# %Y-%M-%D : YYYY-MM-DD
# %H-%N-%S : HH-MM-SS
# %i_%s : ID_vidsecond
# eg: 'cap_2024-05-26_04-20-11_Bpgloy1dDn0_69'


#=============================================================================#
#                                                                             #
#          ███████ ██       █████  ████████ ██████   █████  ██   ██           #
#          ██      ██      ██   ██    ██    ██   ██ ██   ██ ██  ██            #
#          █████   ██      ███████    ██    ██████  ███████ █████             #
#          ██      ██      ██   ██    ██    ██      ██   ██ ██  ██            #
#          ██      ███████ ██   ██    ██    ██      ██   ██ ██   ██           #
#                                                                             #
#=============================================================================#


# Consider adding PPA to get the latest flatpak:
sudo add-apt-repository ppa:flatpak/stable
sudo apt update

# Install flatpak.
sudo apt install -y flatpak

## ALLOWING A FLATPAK PACKAGE ACCESS TO DIR:
sudo flatpak override package_name --filesystem=my_path

# Add the Flathub repo:
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo

#=== Handy flatpak tools
# "Flatseal is a graphical utility to review and modify permissions
#  from your Flatpak applications."
flatpak install flathub com.github.tchx84.Flatseal

# Software Flatpak plugin (OPTIONAL) ❌
# Allows you to install apps without the CLI
########### LOL DONT! Flatpak distributes it's software GUI as a snapd 🤣
# sudo apt install gnome-software-plugin-flatpak

#------------------------

# Dev
# ===

#==== JupyerLab Desktop
# NEVER F'ING WORKS
#flatpak install flathub org.jupyter.JupyterLab

#==== Racket
flatpak install flathub org.racket_lang.Racket

#==== Scratch
flatpak install flathub org.turbowarp.TurboWarp

#==== Zeal
flatpak install flathub org.zealdocs.Zeal
sudo flatpak override --filesystem=home org.zealdocs.Zeal


# Git guis
# --------
# I can never get used to using git through gui, but maybe someday I'll break through.
#==== Gittyup
flatpak install flathub com.github.Murmele.Gittyup

#==== gitg
flatpak install flathub org.gnome.gitg
sudo flatpak override --filesystem=home org.gnome.gitg

# System Stuff
# ============

#==== Filelight (nice visualization of disk space)
flatpak install flathub org.kde.filelight

# Apps
# ====

#==== Evince
flatpak install flathub org.gnome.Evince && \
sudo flatpak override --filesystem=host org.gnome.Evince

#==== Okular
flatpak install flathub org.kde.okular && \
sudo flatpak override --filesystem=host org.kde.okular

#==== Zotero
flatpak install flathub org.zotero.Zotero && \
sudo flatpak override --filesystem=home org.zotero.Zotero

#==== Joplin
flatpak install flathub net.cozic.joplin_desktop && \
sudo flatpak override --filesystem=home net.cozic.joplin_desktop

#==== MarkText
flatpak install flathub com.github.marktext.marktext && \
sudo flatpak override --filesystem=home com.github.marktext.marktext

#==== Obsidian
flatpak install flathub md.obsidian.Obsidian && \
sudo flatpak override --filesystem=home md.obsidian.Obsidian

#==== Logseq
flatpak install flathub com.logseq.Logseq
sudo flatpak override --filesystem=home com.logseq.Logseq


#==== Zettlr
flatpak install flathub com.zettlr.Zettlr

# ----  Media  -------------------------------------------------------------- #

#==== Inkscape
flatpak install flathub org.inkscape.Inkscape && \
sudo flatpak override --filesystem=home org.inkscape.Inkscape

#==== draw.io
flatpak install flathub com.jgraph.drawio.desktop && \
sudo flatpak override --filesystem=home com.jgraph.drawio.desktop

#==== Nomacs
flatpak install flathub org.nomacs.ImageLounge && \
sudo flatpak override --filesystem=home org.nomacs.ImageLounge

#==== Lossless cut (video cut/clip software GUI)
flatpak install flathub no.mifi.losslesscut
sudo flatpak override --filesystem=home no.mifi.losslesscut

#==== Footage (sinple video crop software gui)
flatpak install flathub io.gitlab.adhami3310.Footage


#==== Color-picker
flatpak install flathub nl.hjdskes.gcolor3
sudo flatpak override --filesystem=home nl.hjdskes.gcolor3

#==== Colorway (Color pairings)
flatpak install flathub io.github.lainsce.Colorway

#==== Upscayl
flatpak install flathub org.upscayl.Upscayl

#==== Imaginer (image gen)
flatpak install flathub page.codeberg.Imaginer.Imaginer

# MEDIA PLAYERS
# =============

#==== Moosync (music player; spotify YT etc)
flatpak install flathub app.moosync.moosync

#==== Spot (spotify player)
## NOPE! ONLY SUPPORTS 'PREMIUM'/SUBSCRIBED SPOTIFY ACCOUNTS
#flatpak install flathub dev.alextren.Spot


#==== Spotube (actual spotify client)
# source: https://github.com/KRTirtho/spotube
flatpak install flathub com.github.KRTirtho.Spotube

#==== Monophony (stream music from YT)
flatpak install flathub io.gitlab.zehkira.Monophony

# --------------------------------------------------------------------------- #

#==== This Week in My Life (kanban task organizer)
flatpak install flathub io.github.zhrexl.thisweekinmylife

#==== SaveDesktop (saves gnome settings)
flatpak install flathub io.github.vikdevelop.SaveDesktop
sudo flatpak override --filesystem=home io.github.vikdevelop.SaveDesktop

#==== Emoji-picker: smile
flatpak install flathub it.mijorus.smile
# create keyboard shortcut for: flatpak run it.mijorus.smile

#==== Firmware installer
flatpak install flathub org.gnome.Firmware

#==== Calendar (Gnome)
flatpak install flathub org.gnome.Calendar

#==== Thunderbird
flatpak install flathub org.mozilla.Thunderbird

#==== Blanket (ambient sounds player)
flatpak install flathub com.rafaelmardojai.Blanket

#==== UML & SysML diagramming tool
flatpak install flathub org.gaphor.Gaphor

#==== Hackernews client
flatpak install flathub de.gunibert.Hackgregator

#==== Tangram (a browser with pinned tabs)
flatpak install flathub re.sonny.Tangram

#==== Newsflash (media/blog/rss reader)
flatpak install flathub com.gitlab.newsflash

#==== Standard Notes
flatpak install flathub org.standardnotes.standardnotes

#==== Planner
flatpak install flathub com.github.alainm23.planner

#==== Dev Toolbox (misc lot of handy tools)
flatpak install flathub me.iepure.devtoolbox


# Browsers
# --------
## NOTE: currently do not utilize any of these, so they are commented-out.
# Chromium
#flatpak install flathub org.chromium.Chromium

# Ungoogled Chromium
#flatpak install flathub com.github.Eloston.UngoogledChromium


# Misc
# ----

## TODO STUFF
#==== Done (todo manager)
flatpak install flathub dev.edfloreshz.Done
sudo flatpak override dev.edfloreshz.Done --filesystem=home

#==== Sleek (todo manager)
flatpak install flathub com.github.ransome1.sleek

#==== OpenTodoList (todo manager, todo.txt)
flatpak install flathub net.rpdev.OpenTodoList



# It's There
# ----------
## Stuff that looks interesting hmm..

## Other unused stuff that is situational.

#==== Bottles: run windows stuff
#flatpak install flathub com.usebottles.bottles

#==== Eye-of-gnome (gnome image viewer)
## I think the apt source is probably fine; I don't even use this viewer really.
#flatpak install flathub org.gnome.eog



#=============================================================================#
#                       __  __              _   _                             #
#                      |  \/  |            | | (_)                            #
#                      | \  / |   ___    __| |  _    __ _                     #
#                      | |\/| |  / _ \  /  ` | | |  /  ` |                    #
#                      | |  | | |  __/ | (_| | | | | (_| |                    #
#                      |_|  |_|  \___|  \__,_| |_|  \__,_|                    #
#                                                                             #
#=============================================================================#

#=== Media
# catimg: terminal print (ascii) an image
# gpick: color picker
# mpv: media player for mkv
# nautilus-image-converter: context actions available on right-click image
# sox: terminal audio player
# vlc: media player
# mkvtoolnix: tools for manipulating mkv files
# webp: (dependency that allows support for reading/converting webp image files)
sudo apt install -y catimg gpick mediainfo mpv nautilus-image-converter \
sox vlc mkvtoolnix webp

#=== OBS Studio
# Ref: https://obsproject.com/download
sudo add-apt-repository -y ppa:obsproject/obs-studio && \
sudo apt update && \
sudo apt install -y obs-studio;

#=============================================================================#
#                     '''''''''Social Network''''''''''''                     #
#=============================================================================#

# Twitter client
pip install rainbowstream




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

mkdir -p ~/.cache && cd ~/.cache && \
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

# cheat.sh
# ========
## https://github.com/chubin/cheat.sh
# First, dependencies:
sudo apt install -y rlwrap

# Download to local bin.
curl https://cht.sh/:cht.sh > $HOME/.local/bin/cht.sh
chmod +x $HOME/.local/bin/cht.sh
#----- Note that `cht.sh` is aliased to `cht`.

# WAIT!
# cht.sh has an option `--standalone-install`
#  - so.... is locally downloading the sh file to bin and stuff necessary?
#    - or can you simply use the typical curl usage, and specify local nstall?
## WHATEVER, I did it to bin
## then did, `cht --standalone-install`
##  LOOKS GOOD!






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
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/brave-browser-archive-keyring.gpg] https://brave-browser-apt-release.s3.brave.com/ stable main" | sudo tee /etc/apt/sources.list.d/brave-browser-release.list

# Install.
sudo apt update && sudo apt install -y brave-browser


# Tampermonkey scripts
# ====================
# Pre-fucked GH feed:
https://raw.githubusercontent.com/Gerrit0/old-github-feed/main/old-feed.user.js

# Remove `?tab=readme-ov-file` bullshit:
https://gist.github.com/vogler/74edff6de37c3a13eeff8c99c6bed910/raw/clean-url-readme-tab.github.com.tamper.js


## NB, If you ever see the error message:
# `N: Skipping acquire of configured file 'main/binary-i386/Packages' etc...
# eg:
# `N: Skipping acquire of configured file 'main/binary-i386/Packages' as repository 'https://brave-browser-apt-release.s3.brave.com stable InRelease' doesn't support architecture 'i386'`
## You fix by assigning `arch=amd64` like:
## deb [arch=amd64] https://brave-browser-apt-release.s3.brave.com/ stable main ....ETC

# ^ HAPPENED with Mojo source.
# Fix:
# sudo vi /etc/apt/sources.list.d/modular-installer.list
## deb [arch=amd64 signed-by=/usr/share/keyrings/modular-installer-archive-keyring.gpg] https://dl.modular.com/public/installer/deb/ubuntu jammy main
## deb-src [arch=amd64 signed-by=/usr/share/keyrings/modular-installer-archive-keyring.gpg] https://dl.modular.com/public/installer/deb/ubuntu jammy main


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
#    "word_wrap": false,
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
#    { "keys": ["alt+z"], "command": "toggle_setting", "args": {"setting": "word_wrap"} },
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
#               █████  ██     ████████  ██████   ██████  ██                   #
#              ██   ██ ██        ██    ██    ██ ██    ██ ██                   #
#              ███████ ██        ██    ██    ██ ██    ██ ██                   #
#              ██   ██ ██        ██    ██    ██ ██    ██ ██                   #
#              ██   ██ ██        ██     ██████   ██████  ███████              #
#                                                                             #
#=============================================================================#

# ART/MEDIA
# =========
# Stability-AI StableStudio
# https://github.com/Stability-AI/StableStudio

# Neural MMO
# ==========
# Boy, this one is fussy with package versions.
# It's pinned pip pkg versions are unsatisfiable in my normal env, so I simply
# saw the `setup.py` requirements, installed them all without their deps.
# Good to go brother.

#=== First, get the hdf5 sys stuffs
sudo apt install -y libhdf5-dev hdf5-tools

#=== Install pip packages
pipi --no-deps numpy scipy pytest pytest-benchmark autobahn Twisted vec-noise imageio ordered-set pettingzoo gym pylint psutil py tqdm dill nmmo


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

#=============================================================================#
#                                   WAyland                                   #
#=============================================================================#

# Disable it; the cause of many pains.
sudo vi /etc/gdm3/custom.conf
# Uncomment the line:
#WaylandEnable=False



# Keyboard
# ========
#---- misc/app
flameshot gui    # Print
# Filename:
# screencap_%Y-%m-%d_%H-%M-%S
#   - with png extension

#---- Disable insert
# Figure out what is mapped to insert key
xmodmap -pke | grep -i insert

# Map ins key to null in ~/.Xmodmap
echo "keycode 90 =" >> ~/.Xmodmap

# It's different on Jammy/Wayland.
# Here's the full walkthrough:
#   Ref: https://askubuntu.com/a/1498911

# Install `keyd`
git clone --depth=1 https://github.com/rvaiya/keyd
cd keyd
make && sudo make install
sudo systemctl enable keyd && sudo systemctl start keyd

# Remap the insert key in `/etc/keyd/default.conf`:
#[ids]
#
#*
#
#[main]
#
## Disable insert key.
#insert = noop

# Reload `keyd`:
sudo keyd reload


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
sudo apt install -y arc-theme
# TODO
# Browsing: https://www.gnome-look.org/s/Gnome/browse/

###############################################################################


# CREDITS
# =======
# - jellyfish text art: https://textart.sh/topic/jellyfish
