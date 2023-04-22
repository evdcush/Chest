exit;  # Safety-first!


#=============================================================================#
#                                   Flatpak                                   #
#=============================================================================#

# Install flatpak.
sudo add-apt-repository ppa:flatpak/stable
sudo apt update
sudo apt install -y flatpak

# Add the Flathub repo:
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo



## ALLOWING A FLATPAK PACKAGE ACCESS TO DIR:
sudo flatpak override package_name --filesystem=my_path



#=== Handy flatpak tools
# "Flatseal is a graphical utility to review and modify permissions
#  from your Flatpak applications."
flatpak install flathub com.github.tchx84.Flatseal

# Software Flatpak plugin (OPTIONAL) ❌
# Allows you to install apps without the CLI
########### LOL DONT! Flatpak distributes it's software GUI as a snapd 🤣
# sudo apt install gnome-software-plugin-flatpak

#------------------------

# Apps
# ====

#==== Emoji-picker: smile
flatpak install flathub it.mijorus.smile
# create keyboard shortcut for: flatpak run it.mijorus.smile

#==== Firmware installer
flatpak install flathub org.gnome.Firmware

#==== Nomacs
flatpak install flathub org.nomacs.ImageLounge

#==== Lossless cut (video cut/clip software GUI)
flatpak install flathub no.mifi.losslesscut
sudo flatpak override no.mifi.losslesscut --filesystem=$HOME

#==== Color-picker
flatpak install flathub nl.hjdskes.gcolor3
sudo flatpak override nl.hjdskes.gcolor3 --filesystem=$HOME

#==== Gephi
## > "Gephi is the leading open-source platform for visualizing and manipulating large graphs."
flatpak install flathub org.gephi.Gephi


# Browsers
# --------
## NOTE: currently do not utilize any of these, so they are commented-out.
# Chromium
#flatpak install flathub org.chromium.Chromium

# Ungoogled Chromium
#flatpak install flathub com.github.Eloston.UngoogledChromium


# Misc
# ----
## Other unused stuff that is situational.

#==== Bottles: run windows stuff
#flatpak install flathub com.usebottles.bottles

#==== Eye-of-gnome (gnome image viewer)
## I think the apt source is probably fine; I don't even use this viewer really.
#flatpak install flathub org.gnome.eog



#=============================================================================#
#                       __  __
