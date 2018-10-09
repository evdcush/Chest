###############################################################################
#                                                                             #
#                                                                ,a8888a,     #
#               ,d                                             ,8P"'  `"Y8,   #
#               88                                            ,8P        Y8,  #
#  ,adPPYba,  MM88MMM  ,adPPYYba,   ,adPPYb,d8   ,adPPYba,    88          88  #
#  I8[    ""    88     ""     `Y8  a8"    `Y88  a8P_____88    88          88  #
#   `"Y8ba,     88     ,adPPPPP88  8b       88  8PP"""""""    `8b        d8'  #
#  aa    ]8I    88,    88,    ,88  "8a,   ,d88  "8b,   ,aa     `8ba,  ,ad8'   #
#  `"YbbdP"'    "Y888  `"8bbdP"Y8   `"YbbdP"Y8   `"Ybbd8"'       "Y8888P"     #
#                                   aa,    ,88                                #
#                                    "Y8bbdP"                                 #
#                                                                             #
###############################################################################

#==============================================================================
#                     Preinstallation Steps and Notes                         #
#==============================================================================

# Installation Media
#-----------------------------
#===== Bootable ISO
# Nothing has EVER worked for me in making a bootable USB drive other than
# "Universal USB Installer, pendrive," which, of course, is Windows only

#===== Troubleshooting
# If things are failing with the USB, try a different, non-3.0 port, or
#  not a "hub" port, like the front ones neer power or otherwise joint hubs



# Installer
#-----------------------------
#===== Options
# - DO NOT CONNECT TO INTERNET/DOWNLOAD DURING INSTALL
#   - Remove bloat first
#     ONLY AFTER REMOVING BLOAT, should you even connect to internet
# - Do not install 3rd party shit (for wifi/video/flash etc)
#   - you can install what you need later, and flash 2018 lol



#sed -i "$ a MIRRORS=( 'http://mirror.picosecond.org/ubuntu/,http://mirror.enzu.com/ubuntu/,http://la-mirrors.evowise.com/ubuntu/,http://mirrors.ocf.berkeley.edu/ubuntu/,http://mirror.math.ucdavis.edu/ubuntu/' )" /etc/apt-fast.conf
#MIRRORS=( 'http://mirror.picosecond.org/ubuntu/,http://mirror.enzu.com/ubuntu/,http://la-mirrors.evowise.com/ubuntu/,http://mirrors.ocf.berkeley.edu/ubuntu/,http://mirror.math.ucdavis.edu/ubuntu/' )


# Remove useless shit from default PATH
#--------------------------------------
#===== /usr/games
sudo vi /etc/environment
# delete '/usr/games:/usr/local/games' near the end


###############################################################################
#                                                                             #
#                                                                     88      #
#                 ,d                                                ,d88      #
#                 88                                              888888      #
#    ,adPPYba,  MM88MMM  ,adPPYYba,   ,adPPYb,d8   ,adPPYba,          88      #
#    I8[    ""    88     ""     `Y8  a8"    `Y88  a8P_____88          88      #
#     `"Y8ba,     88     ,adPPPPP88  8b       88  8PP"""""""          88      #
#    aa    ]8I    88,    88,    ,88  "8a,   ,d88  "8b,   ,aa          88      #
#    `"YbbdP"'    "Y888  `"8bbdP"Y8   `"YbbdP"Y8   `"Ybbd8"'          88      #
#                                     aa,    ,88                              #
#                                      "Y8bbdP"                               #
#                                                                             #
###############################################################################

#==============================================================================
#                               Remove Bloat                                  #
#==============================================================================

  # - During this process, all '*desktop' packages will likely get removed.
  #   - this includes: gnome-desktop, mate-desktop, xubuntu-desktop
  #     * DON'T WORRY! These packages are just the bundles of default apps
  #       That come with those respective desktops
  #


# Helpful alias for purging shit
alias arp="sudo apt remove --purge"


# Desktop and Apps
#-----------------------------
# Note: there may be some redundancy
#===== Default desktop shit: Apps, libraries
arp mate-desktop-common mate-calc-common mate-calc mugshot cheese-common libgnome-games-support-common
arp gnome-*
arp "*mate-desktop*"
arp "*firefox*" "transmission-*" "*pidgin*" "*parole*" "*xfburn*" "*libreoffice*"
arp "hplip*" "sgt*" "snapd*" simple-scan "thunderbird*" "*yelp*"


# Fonts
#-----------------------------
# - I like to keep my font directory clean: only the fonts I want
# - The default font set includes font sets for seemingly every language
#   on the planet, and it becomes difficult to find the fonts YOU want
#   and YOU installed.

#===== Purge unwanted and non-English fonts (almost entirely south-Asian btw)
#  Note: this is the COMPLETE LIST
arp fonts-tlwg* fonts-thai* fonts-liberation* fonts-smc* fonts-liberation2* fonts-kacst* fonts-samyak* fonts-lao* fonts-deva* fonts-gujr* fonts-knda* fonts-guru* fonts-navilu* fonts-orya* fonts-beng* fonts-freefont* fonts-khmeros* fonts-mlym* fonts-lklug* fonts-sil* fonts-taml* fonts-sahadeva* fonts-sarai* fonts-lohit* fonts-symbola* fonts-gubbi* fonts-nakula* fonts-telu* fonts-tibetan* fonts-opensymbol* fonts-kalapi* fonts-indic* fonts-gargi* fonts-pagul*

#===== Noto is nice, but it also has about 50 extra languages
#  Instead, I download Noto from source, remove the extra langs from the .db
arp "fonts-noto*"


# Clean
#------------------------------------------------------------------------------
sudo apt autoremove













###############################################################################
#                                                                             #
#                                                                 ad888888b,  #
#                 ,d                                             d8"     "88  #
#                 88                                                     a8P  #
#    ,adPPYba,  MM88MMM  ,adPPYYba,   ,adPPYb,d8   ,adPPYba,          ,d8P"   #
#    I8[    ""    88     ""     `Y8  a8"    `Y88  a8P_____88        a8P"      #
#     `"Y8ba,     88     ,adPPPPP88  8b       88  8PP"""""""      a8P'        #
#    aa    ]8I    88,    88,    ,88  "8a,   ,d88  "8b,   ,aa     d8"          #
#    `"YbbdP"'    "Y888  `"8bbdP"Y8   `"YbbdP"Y8   `"Ybbd8"'     88888888888  #
#                                     aa,    ,88                              #
#                                      "Y8bbdP"                               #
#                                                                             #
###############################################################################

#==============================================================================
#                             Common Dependencies                             #
#==============================================================================

#
python-sphinx texlive-latex-recommended dvipng librsvg2-bin imagemagick docbook2x graphviz

###############################################################################
#                                                                             #
#                                                                             #
#                 ,d                                              ad888888b,  #
#                 88                                             d8"     "88  #
#    ,adPPYba,  MM88MMM  ,adPPYYba,   ,adPPYb,d8   ,adPPYba,             a8P  #
#    I8[    ""    88     ""     `Y8  a8"    `Y88  a8P_____88          aad8"   #
#     `"Y8ba,     88     ,adPPPPP88  8b       88  8PP"""""""          ""Y8,   #
#    aa    ]8I    88,    88,    ,88  "8a,   ,d88  "8b,   ,aa             "8b  #
#    `"YbbdP"'    "Y888  `"8bbdP"Y8   `"YbbdP"Y8   `"Ybbd8"'     Y8,     a88  #
#                                     aa,    ,88                  "Y888888P'  #
#                                      "Y8bbdP"                               #
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
#                                                                       ,d8   #
#                 ,d                                                  ,d888   #
#                 88                                                ,d8" 88   #
#    ,adPPYba,  MM88MMM  ,adPPYYba,   ,adPPYb,d8   ,adPPYba,      ,d8"   88   #
#    I8[    ""    88     ""     `Y8  a8"    `Y88  a8P_____88    ,d8"     88   #
#     `"Y8ba,     88     ,adPPPPP88  8b       88  8PP"""""""    888888888888  #
#    aa    ]8I    88,    88,    ,88  "8a,   ,d88  "8b,   ,aa             88   #
#    `"YbbdP"'    "Y888  `"8bbdP"Y8   `"YbbdP"Y8   `"Ybbd8"'             88   #
#                                     aa,    ,88                              #
#                                      "Y8bbdP"                               #
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
#                                                                8888888888   #
#                 ,d                                             88           #
#                 88                                             88  ____     #
#    ,adPPYba,  MM88MMM  ,adPPYYba,   ,adPPYb,d8   ,adPPYba,     88a8PPPP8b,  #
#    I8[    ""    88     ""     `Y8  a8"    `Y88  a8P_____88     PP"     `8b  #
#     `"Y8ba,     88     ,adPPPPP88  8b       88  8PP"""""""              d8  #
#    aa    ]8I    88,    88,    ,88  "8a,   ,d88  "8b,   ,aa     Y8a     a8P  #
#    `"YbbdP"'    "Y888  `"8bbdP"Y8   `"YbbdP"Y8   `"Ybbd8"'      "Y88888P"   #
#                                     aa,    ,88                              #
#                                      "Y8bbdP"                               #
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
#                                                                  ad8888ba,  #
#                 ,d                                              8P'    "Y8  #
#                 88                                             d8           #
#    ,adPPYba,  MM88MMM  ,adPPYYba,   ,adPPYb,d8   ,adPPYba,     88,dd888bb,  #
#    I8[    ""    88     ""     `Y8  a8"    `Y88  a8P_____88     88P'    `8b  #
#     `"Y8ba,     88     ,adPPPPP88  8b       88  8PP"""""""     88       d8  #
#    aa    ]8I    88,    88,    ,88  "8a,   ,d88  "8b,   ,aa     88a     a8P  #
#    `"YbbdP"'    "Y888  `"8bbdP"Y8   `"YbbdP"Y8   `"Ybbd8"'      "Y88888P"   #
#                                     aa,    ,88                              #
#                                      "Y8bbdP"                               #
#                                                                             #
###############################################################################

























#      888888888888       ad88888ba
#              ,8P'      d8"     "8b
#             d8"        Y8a     a8P
#           ,8P'          "Y8aaa8P"
#          d8"            ,d8"""8b,
#        ,8P'            d8"     "8b
#       d8"              Y8a     a8P
#      8P'                "Y88888P"
