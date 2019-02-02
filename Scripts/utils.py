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

# Primary dests
# =============
DOTS  = f"{HOME}/.Dots"
APPS  = f"{HOME}/.Apps"


# Local
# =====
BIN      = f"{HOME}/.local/bin"
DESKTOPS = f"{HOME}/.local/share/applications"


# Homedirs
# ========
CHEST = f"{HOME}/Chest"
CLOUD = f"{HOME}/Cloud"
MEDIA = f"{HOME}/Media"
PROJECTS  = f"{HOME}/Projects"
DOCUMENTS = f"{HOME}/Documents"


#==== .apps
ICONS = f"{APPS}/Icons"
NATIVEFIED = f"{APPS}/Nativefied"
BINARIES   = f"{APPS}/Binaries"

#==== CLOUD
READMES = f"{CLOUD}/READMEs"
RESOURCES_CLOUD = f"{CLOUD}/Resources"
DOCUMENTS_CLOUD = f"{CLOUD}/Reading"

#==== chest
DOTS_CHEST = f"{CHEST}/Dots"
RESOURCES_CHEST = f"{CHEST}/Resources"

#==== Projects
HOARD = f"{PROJECTS}/Hoard/Archive"
PAPER = f"{PROJECTS}/DocHub/Literature"
PREP  = f"{PROJECTS}/Prep"


#=============================================================================#
#                            __   _   _                                       #
#                           / _| (_) | |   ___   ___                          #
#                          | |_  | | | |  / _ \ / __|                         #
#                          |  _| | | | | |  __/ \__ \                         #
#                          |_|   |_| |_|  \___| |___/                         #
#                                                                             #
#=============================================================================#

# fpaths
inbox_hoard_path   = f"{RESOURCES_CLOUD}/inbox_hoard.yml"
archive_hoard_path = f"{RESOURCES_CLOUD}/archive_hoard.yml"
inbox_read_path    = f"{CLOUD}/Reading/inbox.txt"

#==== resources
GH_HOARD  = R_yml(f"{RESOURCES_CLOUD}/inbox_hoard.yml")
HOARD_ARC = R_yml(f"{RESOURCES_CLOUD}/archive_hoard.yml")
PUBLIC_TOKENS = R_yml(f"{RESOURCES_CLOUD}/gh_tokens.yml")['public']

#==== projects
READ_INBOX = open(inbox_read_path, 'a')
+        txt_len = len(rtxt) + 4
+        W = W if W > txt_len else txt_len
