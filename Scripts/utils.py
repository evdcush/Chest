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
AWESOME_LISTS = READMES + "/awesome_lists"
RESOURCES_CLOUD = f"{CLOUD}/Resources"
DOCUMENTS_CLOUD = f"{CLOUD}/Reading"

#==== chest
DOTS_CHEST = f"{CHEST}/Dots"
RESOURCES_CHEST = f"{CHEST}/Resources"

#==== Projects
HOARD = f"{PROJECTS}/Hoard"
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
inbox_read_path    = f"{CLOUD}/Reading/inbox.txt"

#==== resources
#HOARD_INBOX   = R_yml(f"{RESOURCES_CLOUD}/inbox_hoard.yml")
#HOARD_ARCHIVE = R_yml(f"{RESOURCES_CLOUD}/archive_hoard.yml")
PUBLIC_TOKENS = R_yml(f"{RESOURCES_CLOUD}/gh_tokens.yml")['public']

def add_to_read_inbox(entry):
    with open(inbox_read_path, 'a') as ibx:
        ibx.write(entry + '\n')

#=============================================================================#
#           _                                  _   _                          #
#          | |__     ___     __ _   _ __    __| | (_)  _ __     __ _          #
#          | '_ \   / _ \   / _` | | '__|  / _` | | | | '_ \   / _` |         #
#          | | | | | (_) | | (_| | | |    | (_| | | | | | | | | (_| |         #
#          |_| |_|  \___/   \__,_| |_|     \__,_| |_| |_| |_|  \__, |         #
#                                                              |___/          #
#                                                                             #
#=============================================================================#


#-----------------------------------------------------------------------------#
#                                 Hoard files                                 #
#-----------------------------------------------------------------------------#
# Paths
hoard_inbox_path   = f"{HOARD}/inbox.yml"
hoard_archive_path = f"{HOARD}/archive.yml"

# Inbox/archive mapping
hoard_inbox_keys = {'o': 'orgs', 'r': 'repos', 'u':'users'}

# smelly mapping: 'files' keys are mapped in somewhat opposite manner
#  to inbox keys, but it makes the archiving func a little cleaner
hoard_archive_keys = {'awesome_lists': 'files',
                      'readmes': 'files',
                      'src': 'files',
                      'gists': 'files',
                      **hoard_inbox_keys}

#-----------------------------------------------------------------------------#
#                               hoard rw utils                                #
#-----------------------------------------------------------------------------#

# Hoard inboxing
def add_to_inbox(key, entry):
    assert key in hoard_inbox_keys
    key = hoard_inbox_keys[key]
    hoard = R_yml(hoard_inbox_path)
    hoard[key].append(entry)
    W_yml(hoard_inbox_path, hoard)

# Hoard archive
def add_to_archive(key, entry):
    """
    Two types of indexing
    add_to_archive('src', <entry>)
      key == 'src'; hoard_key == 'files'
      hoard_archive[hoard_key][key] == hoard_archive['files']['src']

    add_to_archive('r', <entry>)
      key == 'r'; hoard_key == 'repos';
      hoard_archive[hoard_key]

    """
    assert key in hoard_archive_keys
    hoard_key = hoard_archive_keys[key]
    hoard_archive = R_yml(hoard_archive_path)

    if hoard_key == 'files':
        hoard_archive[hoard_key][key].append(entry)
    else:
        hoard_archive[hoard_key].append(entry)
    W_yml(hoard_archive_path, hoard_archive)

