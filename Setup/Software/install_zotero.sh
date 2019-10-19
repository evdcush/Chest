#!/bin/bash

# Zotero installer
#   - scrape zotero page for current download version
#   - download latest version for linux-x86_64
#   -
#  Based on github.com/smathot/zotero_installer
#
#
#  ASSUMES : version info contained within one line, and can be found
#            at ret[-11:-6] (of format X.X.XX)


#=== Get Latest version
VER=`wget -q -O- https://www.zotero.org/download | grep 'linux-x86_64' | rev | cut -c 6-11 | rev`

#=== Format download URL
TARGET="https://www.zotero.org/download/client/dl?channel=release&platform=linux-x86_64&version=$VER"
echo "Download URL: $TARGET"

# make cache if not exist
mkdir -p ~/.cache;

# Download



#=== Download zotero and unarchive to local share
wget $TARGET -O ~/.cache/zotero.tar.bz2
tar -xpf ~/.cache/zotero.tar.bz2 -C ~/.local/share/zotero
mv ~/.local/share/{Zotero_linux-x86_64,zotero}

#=== Make desktop entry, symlink files
echo 'Creating zotero.desktop; linking files'
echo "[Desktop Entry]
Name=Zotero
Comment=Open-source reference manager
Exec=$USER/.local/bin/zotero
Icon=$USER/.local/share/zotero/chrome/icons/default/default256.png
Type=Application
StartupNotify=true" > ~/.local/share/applications

ln -sf $USER/.local/zotero/zotero $USER/.local/bin

echo 'FINISHED!'


#https://www.zotero.org/download/client/dl?channel=release&platform=linux-x86_64&version=5.0.75
#https://www.zotero.org/download/client/dl?channel=release&platform=linux-x86_64&version=5.0.75
