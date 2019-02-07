#!/usr/bin/env python

""" Automate most of the nativefier process
"""
import os
import subprocess

# import favicon  <--------<<<< Library to find favicon from a site
#   it would be far simpler if you could scrape the favicon, svg --> png it, and use that

from argparse import ArgumentParser
from slugify import slugify


#==============================================================================
# Constants
#    pathing and formatting mostly
#==============================================================================

# Constants
#------------------
#==== Root dirs
#PWD   = os.environ["PWD"] # default installation dir
HOME  = os.environ["HOME"]
LOCAL = f'{HOME}/.local'
#==== Dest dirs
LOCAL_BIN  = f'{LOCAL}/bin'                 # for symlinked binary
LOCAL_APPS = f'{LOCAL}/share/applications'  # for `<app>.desktop` file
DEFAULT_ARGS = '--file-download-options \'{"saveAs": true}\' -m'
#==== User-specific paths
APP_PATH     = f'{HOME}/.Apps/nativefied'    # app source dir
DESKTOP_PATH = APP_PATH + '/desktop_entries' # desktop files to symlink
ICON_DIR  = f'{HOME}/.Apps/Icons'       # personally sourced icon assets
ICON_PATH = f'{ICON_DIR}/{{name}}.png'

# Desktop entry format
DESKTOP_ENTRY = f"""
[Desktop Entry]
Name={{title}}
Comment=Nativefied {{title}}
Exec={LOCAL_BIN}/{{name}}
Terminal=false
Type=Application
Icon={{path}}/resources/app/icon.png
Categories=Network;
"""
#==============================================================================
# Argparser -
#    supports subset of nativefier flags
#==============================================================================

# Argparser
#------------------
P = ArgumentParser()
adg = P.add_argument
#==== parser args
adg('url', type=str,
    help='url of target webapp')

adg('-n','--name', type=str, required=True,
    help='name of the app; determines directory name as well as binary')

adg('-i', '--icon', type=str,
    help='.png icon for app')

adg('-p', '--path', type=str, default=APP_PATH,
    help='full-path to installation directory, (default="~/.Apps")')

adg('-u', '--internal-urls', type=int,
    help="apparently regex? I used int before and that worked")
# nativefier ex: --internal-urls ".*?\.google\.*?" (but I just use num?)

adg('-t', '--tray', action='store_true',
    help='whether the app remains in tray when closed')

adg('-s', '--single-instance', action='store_false',
    help='only allow single instance of app')

adg('-c', '--counter', action='store_true',
    help='X number attached to window label for apps that support count, such as gmail')


# Parse user args
#----------------
def parse_args():
    """ Interprets argparse args for nativefier commands
    """
    args = vars(P.parse_args())
    name = args['name']
    url  = args['url']
    print(f'parse_args: name = {name}; url = {url}')

    # Helper
    # ------
    def _decode_flag(key):
        # manages boolean flag args
        val = args[key]
        if val:
            args[key] = f'--{key}'
        else:
            args.pop(key, None)

    # Bool flags
    # ----------
    _decode_flag('tray')
    _decode_flag('counter')
    _decode_flag('single_instance')

    # Interpreted args
    # ----------------
    #==== icon
    if args['icon'] is None:
        icon_name = name.lower()
        icon_path = ICON_PATH.format(name=icon_name)
        args['icon'] = f'-i {icon_path}'
    #==== internal-urls
    if args['internal_urls'] is None:
        args.pop('internal_urls', None)
    else:
        internal_urls = args['internal_urls']
        args['internal_urls'] = f'--internal_urls {internal_urls}'
    #==== name
    args['name'] = f'-n {name}'
    return args

#=============================================================================#
# Nativefy :                                                                  #
#   1. build nativefied app                                                   #
#   2. make app desktop entry                                                 #
#   3. symlink files:                                                         #
#       ln -sf <app_binary>  <local bin>                                      #
#       ln -sf <app_desktop> <local applications>                             #
#=============================================================================#

# File and pathing utils
# ======================
def mkdirs(p):
    if not os.path.exists(p):
        os.makedirs(p)

def symlink(from_path, to_path=LOCAL_BIN):
    subprocess.run(f'ln -sf {from_path} {to_path}', shell=True)
    print(f'\nsymlinked -\nFROM: {from_path}\n  TO: {to_path}')

def make_binary_exec(bin_file_path):
    subprocess.run(f'chmod +x {bin_file_path}', shell=True)
    print('\nMade binary executable')

def make_desktop_entry(name, slug_name, app_path):
    app_title = name if name[0].isupper() else name.title()
    file_name = f'{name}.desktop'
    de = DESKTOP_ENTRY.format(title=app_title, name=slug_name, path=app_path)
    mkdirs(DESKTOP_PATH)
    de_path = f"{DESKTOP_PATH}/{file_name}"
    with open(de_path, 'w') as entry:
        entry.write(de)
        symlink(de_path, LOCAL_APPS)

# Main interface
# ==============
def nativefy():
    """ Make desktop app using nativefier

    Order of ops
    ------------
    1. build nativefier command
        a. parse user args from STDIN
        b. format args to valid nativefier args
    2. call nativefier
    3. make app binaries executable
        a. symlink binaries to local bin
    5. make desktop entry
        b. mv .desktop file to local apps

    Params
    ------
    opts : dict
        the parsed args, contains all relevant nativefier flags
    """
    opts = parse_args()
    name = opts['name'].split(' ')[-1]
    nativefier_commands = []
    add_arg = lambda k: nativefier_commands.append(opts.pop(k))

    # Targets
    # -------
    add_arg('url')
    add_arg('name')

    # Destination
    # -----------
    dest = opts['path']
    if not os.path.exists(dest): os.makedirs(dest)
    opts.pop('path', None) # remove non-native cmd

    # Default args
    nativefier_commands.append(DEFAULT_ARGS)

    # Variable args
    # -------------
    var_args = dict(opts)
    for k in var_args: add_arg(k)

    # Format nativefier command string
    # --------------------------------
    nativefier_cmd = ' '.join(nativefier_commands)
    nativefier_cmd = 'nativefier ' + nativefier_cmd

    # Execute
    # -------
    print(nativefier_cmd)
    subprocess.run(nativefier_cmd, cwd=dest, shell=True)
    print('\nFinished nativefier command')

    # Link binaries
    # -------------
    slug_name = slugify(name)
    app_file_path = f'{dest}/{slug_name}-linux-x64'
    bin_file_path = f'{app_file_path}/{slug_name}'
    make_binary_exec(bin_file_path)
    symlink(bin_file_path)

    # Make desktop entry
    make_desktop_entry(name, slug_name, app_file_path)
    print('Finished nativefication process!')


# Nativefy a web app
# ------------------
if __name__ == '__main__':
    nativefy()
