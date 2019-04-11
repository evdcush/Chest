import os
import sys
import code
import argparse
from functools import wraps
#import yaml
from ruamel.yaml import YAML
yaml = YAML()

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

#-----------------------------------------------------------------------------#
#                                   Parser                                    #
#-----------------------------------------------------------------------------#

# Parser
# ======
CLI = argparse.ArgumentParser()

# Subcommands
# ===========
def subcmd(*args, **kwargs):
    """Decorator to define a new subcommand in a sanity-preserving way.

    The function will be stored in the ``func`` variable when the parser
    parses arguments so that it can be called directly like so::
        args = cli.parse_args()
        args.func(args)

    Usage example::
        @subcmd("-d", help="Enable debug mode", action="store_true")
        def foo(args):
            print(args)

    Then on the command line::
        $ python cli.py foo -d
    """
    global subparsers
    if subparsers is None:
        subparsers = cli.add_subparsers(dest='subcmd')
    parent = subparsers
    def decorator(func):
        parser = parent.add_parser(func.__name__, description=func.__doc__)
        #for args, kwargs in parser_args:
        #    parser.add_argument(*args, **kwargs)
        parser.add_argument(*args, **kwargs)
        parser.set_defaults(func=func)
    return decorator


"""
# SAMPLES
@subcmd()  # INVALID with updated subcmd
def nothing(args):
    print("Nothing special!")

@subcmd('-d', help='debug mod', action='store_true')
def test(args):
    print(args)

@subcmd('-f', '--filename', help="A thing with a filename")
def filename(args):
    print(args.filename)

@subcmd('name', help='Name')
def name(args):
    print(args.name)
"""



#-----------------------------------------------------------------------------#
#                                Handy Helpers                                #
#-----------------------------------------------------------------------------#

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


#-----------------------------------------------------------------------------#
#                                    PATHS                                    #
#-----------------------------------------------------------------------------#

### DIRS

# user root
HOME  = os.environ['HOME']

# Dot dirs
# ========
DOTS = HOME + '/.Dots'
APPS = HOME + '/.Apps'
BIN  = HOME + '/.local/bin'
DESKTOPS = HOME + '/.local/share/applications'

# Home dirs
# =========
CHEST     = HOME + '/Chest'
CLOUD     = HOME + '/Cloud'
PROJECTS  = HOME + '/Projects'
DOCUMENTS = HOME + '/Documents'

# Projects
HOARD = PROJECTS + '/Hoard'
HOARD_ARCHIVE = HOARD + '/Archive'
PAPER = PROJECTS + '/DocHub/Literature'

# Apps
ICONS      = APPS + '/Icons'
BINARIES   = APPS + '/Binaries'
NATIVEFIED = APPS + '/Nativefied'

# Cloud dirs
# ==========
READMES = CLOUD + '/READMEs'
AWESOME_LISTS = READMES + "/awesome_lists"
CLOUD_RESOURCES = CLOUD + '/Resources'
CLOUD_READING   = CLOUD + '/Reading'

# Chest dirs
# ==========
CHEST_DOTS      = CHEST + '/Dots'
CHEST_RESOURCES = CHEST + '/Resources'
CHEST_TEMPLATES = CHEST_RESOURCES + '/Templates'
CHEST_LICENSES  = CHEST_TEMPLATES + '/Licenses'


### FILES

# API tokens
# ==========
__tokens_path = DOTS + '/tokens.yml'
_tokens = lambda: R_yml(__tokens_path)
TOKEN_KAGGLE = lambda: _tokens()['kaggle']
TOKEN_GITHUB = lambda: _tokens()['github']
TOKEN_ZOTERO = lambda: _tokens()['zotero']


# Plain text licenses
# ===================
LICENSES = AttrDict(
    bsd3=CHEST_LICENSES + '/BSD-3-Clause.txt',
    lgpl=CHEST_LICENSES + '/LGPL-3.0.txt',
    agpl=CHEST_LICENSES + '/AGPL-3.0.txt',
    ccsa=CHEST_LICENSES + '/CC-BY-NC-SA-4.0.txt',
    ccnd=CHEST_LICENSES + '/CC-BY-NC-ND-4.0.txt',
    )

##inbox_read_path = f"{CLOUD}/Reading/inbox.txt"  # deprecated, dochub in use
#def add_to_read_inbox(entry):
#    with open(inbox_read_path, 'a') as ibx:
#        ibx.write(entry + '\n')

#-----------------------------------------------------------------------------#
#                                  Hoarding                                   #
#-----------------------------------------------------------------------------#

# Hoard vars
# ==========
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

# Hoard funcs
# ===========
def init_github():
    """ initialize github sess """
    import github3
    tokens = TOKEN_GITHUB()
    hoard_token = tokens['hoard']['token']
    github = github3.login(token=hoard_token)
    return github


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


#-----------------------------------------------------------------------------#
#                                    MISC                                     #
#-----------------------------------------------------------------------------#


#=== Chars for copy
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
