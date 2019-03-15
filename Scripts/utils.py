import os
import sys
import code
from functools import wraps
import yaml

#-----------------------------------------------------------------------------#
#                                Handy Helpers                                #
#-----------------------------------------------------------------------------#

""" Unicode math symbols
₀₁₂₃₄₅₆₇₈₉
𝑤𝑢𝑣𝑥𝑦𝑧𝑠𝑡𝑝𝑞𝑛𝑚𝑓𝑔𝑖𝑗𝑘
∘
≠
≈
∈
→
←
±
≤≥
∑
𝛁
𝜕
∆
ϕ
ϴ

†
‡

⍺
𝜶𝜶𝛼𝛂

𝑛
𝒏

𝝺
𝛌
𝝀𝜆
∑∴≤≥≠≈⊧⊦⟮⟯⨀⨁⨂⫤𝛁∊∆→←±÷×+|
϶∘∙²³¹ϐ₊₋ₓ

"""

# yaml rw helpers
# ===============
def R_yml(fname):
    with open(fname) as file:
        return yaml.load(file)

def W_yml(fname, obj):
    with open(fname, 'w') as file:
        yaml.dump(obj, file, default_flow_style=False)


# Decorators
# ==========
def NOTIMPLEMENTED(f):
    """ Like TODO, but for functions in a class
        raises error when wrappee is called """
    @wraps(f)
    def not_implemented(*args, **kwargs):
        func_class = args[0]
        f_class_name = func_class.get_class_name()
        f_name = f.__name__
        msg = '\n  Class: {}, function: {} has not been implemented!\n'
        print(msg.format(f_class_name, f_name))
        raise NotImplementedError
    return not_implemented



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

# Dot dirs
# ========
DOTS = f"{HOME}/.Dots"
APPS = f"{HOME}/.Apps"
BIN  = f"{HOME}/.local/bin"
DESKTOPS = f"{HOME}/.local/share/applications"

#-----------------------------------------------------------------------------#
#                                 Home Paths                                  #
#-----------------------------------------------------------------------------#
#==== Home parents
CHEST = f"{HOME}/Chest"
CLOUD = f"{HOME}/Cloud"
PROJECTS  = f"{HOME}/Projects"
DOCUMENTS = f"{HOME}/Documents"

#==== .apps
ICONS = f"{APPS}/Icons"
NATIVEFIED = f"{APPS}/Nativefied"
BINARIES   = f"{APPS}/Binaries"

#==== Cloud
READMES = f"{CLOUD}/READMEs"
AWESOME_LISTS = READMES + "/awesome_lists"
RESOURCES_CLOUD = f"{CLOUD}/Resources"
DOCUMENTS_CLOUD = f"{CLOUD}/Reading"

#==== Chest
DOTS_CHEST = f"{CHEST}/Dots"
RESOURCES_CHEST = f"{CHEST}/Resources"

#==== Projects
HOARD = f"{PROJECTS}/Hoard"
HOARD_ARCHIVE = HOARD + "/Archive"
PAPER = f"{PROJECTS}/DocHub/Literature"
PREP  = f"{PROJECTS}/Prep"

# Files
# =====

# CURRENTLY UNUSED
# ================
"""
MEDIA = f"{HOME}/Media"

"""
#=============================================================================#
#                            __   _   _                                       #
#                           / _| (_) | |   ___   ___                          #
#                          | |_  | | | |  / _ \ / __|                         #
#                          |  _| | | | | |  __/ \__ \                         #
#                          |_|   |_| |_|  \___| |___/                         #
#                                                                             #
#=============================================================================#

# fpaths
inbox_read_path = f"{CLOUD}/Reading/inbox.txt"

#==== resources
api_tokens_path = f"{RESOURCES_CLOUD}/api_tokens.yml"
zot_token = lambda: R_yml(api_tokens_path)['zotero']
public_gh_tokens = lambda: R_yml(api_tokens_path)['github']

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
def init_github():
    import github3
    tokens = public_gh_tokens()
    hoard_token = tokens['hoard']['token']
    github = github3.login(token=hoard_token)
    return github


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
        #code.interact(local=dict(globals(), **locals()))
        hoard_archive[hoard_key][key].append(entry)
    else:
        hoard_archive[hoard_key].append(entry)
    W_yml(hoard_archive_path, hoard_archive)

