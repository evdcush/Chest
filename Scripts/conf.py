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


#                 _     _
#   _ __   __ _  | |_  | |_    ___
#  | '_ \ / _` | |  _| | ' \  (_-<
#  | .__/ \__,_|  \__| |_||_| /__/
#  |_|

HOME  = os.environ['HOME']

###  BASE LEVEL  ###
#==== hidden
LOCAL = f"{HOME}/.local"
NSYNC = f"{HOME}/.NSync"
DOTS  = f"{HOME}/.Dots"
APPS  = f"{HOME}/.Apps"

#==== home dirs
CHEST = f"{HOME}/Chest"
MEDIA = f"{HOME}/Media"
PROJECTS = f"{HOME}/Projects"
DOCUMENTS = f"{HOME}/Documents"


###  SUBDIRS  ###
#==== .local
BIN = f"{LOCAL}/bin"
DESKTOPS = f"{LOCAL}/share/applications"

#==== .apps
NATIVEFIED = f"{APPS}/Nativefied"
BINARIES   = f"{APPS}/Binaries"
ICONS = F"{APPS}/Icons"

#==== .nsync
RESOURCES_N = f"{NSYNC}/Resources"
READMES = f"{NSYNC}/READMEs"
DOCUMENTS_N = f"{NSYNC}/Documents"


#==== chest
RESOURCES_C = f"{CHEST}/Resources"
DOTS_C = f"{CHEST}/Dots"

#==== Projects
HOARD = f"{PROJECTS}/Hoard/Archive"
PAPER = f"{PROJECTS}/PaperNotes/Literature"
PREP  = f"{PROJECTS}/Prep"


#   __   _   _
#  / _| (_) | |  ___   ___
# |  _| | | | | / -_) (_-<
# |_|   |_| |_| \___| /__/
#

#==== resources
PUBLIC_TOKENS = R_yml(f"{RESOURCES_N}/gh_tokens.yml")['public']
GH_HOARD  = R_yml(f"{RESOURCES_N}/inbox_hoard.yml")
HOARD_ARC = R_yml(f"{RESOURCES_N}/archive_hoard.yml")

#==== projects
BIB = f"{DOCUMENTS_N}/library.bib"
