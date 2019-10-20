#!/bin/bash

###############################################################################
#                             Zotero installer
#   - get latest zotero version from zotero download page response
#   - download latest version for linux-x86_64
#   - install zotero for user
#
#  ** Based on github.com/smathot/zotero_installer **
#
#
#  ASSUMES : version info contained within one line, and can be found
#            at ret[-11:-6] (of format X.X.XX)
#
###############################################################################


# =================== #
# GET DOWNLOAD TARGET #
# =================== #

ZDL='https://www.zotero.org/download'  # zotero download splash
PLAT='linux-x86_64'                    # 'mac', 'win32', 'linux-i686' available

#=== Get Latest version
VER=`wget -q -O- $ZDL | grep $PLAT | tail -c 12 | cut -d'"' -f1`

# ALTERNATIVELY
#VER=`wget -q -O- https://www.zotero.org/download \
#| grep 'linux-x86_64' | rev | cut -c 6-11 | rev`


#=== Format download URL
TARGET="$ZDL/client/dl?channel=release&platform=$PLAT&version=$VER"
echo "    Download URL: $TARGET"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# ======== #
# DOWNLOAD #
# ======== #

#=== Download
mkdir -p ~/.cache;                        # mk dest if not exist
wget $TARGET -O ~/.cache/zotero.tar.bz2   # download zotero
echo '    Download complete!'

#=== Extract archive
tar -xpf ~/.cache/zotero.tar.bz2 -C ~/.local/share   # extract to user share
mv ~/.local/share/{Zotero_linux-x86_64,zotero}       # rename
echo '    Installing zotero to ~/.local/share/zotero ...'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# ============== #
# SETUP FOR USER #
# ============== #

echo '    Creating menu entry for zotero ...'

#=== Make desktop entry (for menu access)
mkdir -p ~/.local/share/applications
echo 'Creating zotero.desktop; linking files'
echo "[Desktop Entry]
Name=Zotero
Comment=Open-source reference manager
Exec=$HOME/.local/bin/zotero
Icon=$HOME/.local/share/zotero/chrome/icons/default/default256.png
Type=Application
StartupNotify=true" > ~/.local/share/applications/zotero.desktop

echo '    Menu entry created at ~/.local/share/applications/zotero.desktop'
echo '    Linking binary ...'

#=== Link binary
mkdir -p ~/.local/bin  # in case no user bin
ln -sf $HOME/.local/share/zotero/zotero $HOME/.local/bin  # abs paths used

echo '    FINISHED!'



###############################################################################

# Notes on version part
# ---------------------
#
# There may be a cleaner way, but the quickest and dirtiest was just
#   to wget the zotero downloads page, grep for the relevant build,
#   and extract the version info with some hacky core utils
#
#   This is a sample return from the wget and grep:
#
#   $ wget -q -O- https://www.zotero.org/download | grep 'linux-x86_64'
#               React.createElement(ZoteroWebComponents.Downloads, {"standaloneVersions":{"mac":"5.0.76","win32":"5.0.76","linux-i686":"5.0.76","linux-x86_64":"5.0.76"}}),
#
#   So you can get version info from any build, use whatever
#     utils you like, probably ideal for awk or sed, and extract the ver info.
#     Both methods I chose are 'hardcoded' solutions, which assume
#     the grepped line is static, as is the versioning syntax (X.X.XX)

