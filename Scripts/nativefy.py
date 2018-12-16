#!/usr/bin/env python

""" Automate most of the nativefier process
"""
import os
import subprocess

# import favicon  <--------<<<< Library to find favicon from a site
#   it would be far simpler if you could scrape the favicon, svg --> png it, and use that

from argparse import ArgumentParser


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
APP_PATH  = f'{HOME}/.Apps/nativefied'        # app source dir
ICON_DIR  = f'{HOME}/.Apps/Icons' # personally sourced icon assets
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


def nativefier_slug_workaround(name):
    """ nativefier splits camelCased names with a dash '-'
    which is unfortunate for camelCased services
    """
    if name.istitle() or name.islower() or name.isupper() or '-' in name:
        return name.lower()
    else:
        upper_range = range(ord('A'), ord('Z') + 1)
        lower_range = range(ord('a'), ord('z') + 1)
        slug = ''
        for i, c in enumerate(name[:-1]):
            if ord(c) in lower_range and ord(name[i+1]) in upper_range:
                slug += c + '-'
                continue
            slug += c
        slug += name[-1]
        return slug.lower()

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

#==============================================================================
# Nativefy :
#   1. build nativefied app
#   2. make app desktop entry
#      a. mv .desktop to .local/share/applications
#   3. symlink binary to local bin
#==============================================================================

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
    slug_name = get_dir_name_from_nativefiers_weird_ass_slug_logic(name)
    app_file_path = f'{dest}/{slug_name}-linux-x64'
    bin_file_path = f'{app_file_path}/{slug_name}'
    make_binary_exec(bin_file_path)
    symlink_binary(bin_file_path)

    # Make desktop entry
    make_desktop_entry(name, slug_name, app_file_path)
    print('Finished nativefication process!')


def make_desktop_entry(name, slug_name, app_path):
    app_title = name if name[0].isupper() else name.title()
    file_name = f'{name}.desktop'
    de = DESKTOP_ENTRY.format(title=app_title, name=slug_name, path=app_path)
    with open(f'{LOCAL_APPS}/{file_name}', 'w') as entry:
        entry.write(de)
        print(f'\n{file_name} written to {LOCAL_APPS}')

def symlink_binary(from_path, to_path=LOCAL_BIN):
    subprocess.run(f'ln -sf {from_path} {to_path}', shell=True)
    print(f'\nsymlinked binary -\nFROM: {from_path}\n  TO: {to_path}')

def make_binary_exec(bin_file_path):
    subprocess.run(f'chmod +x {bin_file_path}', shell=True)
    print('\nMade binary executable')


# Nativefy a web app
# ------------------
if __name__ == '__main__':
    nativefy()
