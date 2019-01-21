import os
import sys
import yaml

# File RW helpers
# ===============
def R_yml(fname):
    with open(fname) as file:
        return yaml.load(file)

def W_yml(fname, obj):
    with open(fname, 'w') as file:
        yaml.dump(obj, file, default_flow_style=False)


#=============================================================================#
#                                       _     _                               #
#                       _ __     __ _  | |_  | |__    ___                     #
#                      | '_ \   / _` | | __| | '_ \  / __|                    #
#                      | |_) | | (_| | | |_  | | | | \__ \                    #
#                      | .__/   \__,_|  \__| |_| |_| |___/                    #
#                      |_|                                                    #
#                                                                             #
#=============================================================================#

###  BASE LEVEL  ###
HOME  = os.environ['HOME']

#==== hidden
LOCAL = f"{HOME}/.local"
DOTS  = f"{HOME}/.Dots"
APPS  = f"{HOME}/.Apps"

#==== home dirs
CHEST = f"{HOME}/Chest"
CLOUD = f"{HOME}/Cloud"
MEDIA = f"{HOME}/Media"
PROJECTS  = f"{HOME}/Projects"
DOCUMENTS = f"{HOME}/Documents"


###  SUBDIRS  ###
#==== .local
BIN = f"{LOCAL}/bin"
DESKTOPS = f"{LOCAL}/share/applications"

#==== .apps
ICONS = F"{APPS}/Icons"
NATIVEFIED = f"{APPS}/Nativefied"
BINARIES   = f"{APPS}/Binaries"


#==== CLOUD
READMES = f"{CLOUD}/READMEs"
RESOURCES_CLOUD = f"{CLOUD}/Resources"
DOCUMENTS_CLOUD = f"{CLOUD}/Documents"


#==== chest
DOTS_CHEST = f"{CHEST}/Dots"
RESOURCES_CHEST = f"{CHEST}/Resources"


#==== Projects
HOARD = f"{PROJECTS}/Hoard/Archive"
PAPER = f"{PROJECTS}/PaperNotes/Literature"
PREP  = f"{PROJECTS}/Prep"


#=============================================================================#
#                            __   _   _                                       #
#                           / _| (_) | |   ___   ___                          #
#                          | |_  | | | |  / _ \ / __|                         #
#                          |  _| | | | | |  __/ \__ \                         #
#                          |_|   |_| |_|  \___| |___/                         #
#                                                                             #
#=============================================================================#

#==== resources
GH_HOARD  = R_yml(f"{RESOURCES_N}/inbox_hoard.yml")
HOARD_ARC = R_yml(f"{RESOURCES_N}/archive_hoard.yml")
PUBLIC_TOKENS = R_yml(f"{RESOURCES_N}/gh_tokens.yml")['public']

#==== projects
BIB = f"{DOCUMENTS_N}/library.bib"
