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
