""" Automate most of the nativefier process

# Why?
It's easy enough to nativefy something, but having to explicitly path
certain things and make desktop entries is tedious work.

# What is interpreted/automated?
- Most of the pathing is handled by the script: the app save dir, the binary
  path, the desktop entry path
  - NOTE: change the defaults to your specs (especially icons dir)

- Most of the core nativefier options are supported and simplified here,
  which, in combination with the pathing, significantly reduces the length of
  a normal nativefier call

- Automatically generates a default desktop entry for the nativefied app,
  and puts it in the correct location

# ASSUMED:
- you have adjusted the constants in the script to your setup
- you have the icon (png) for the app of interest
#===== Icon
- You must download/prepare the icon used before calling on this script.
- Icon name must be formatted to match 'appname.png'--the single-word,
  lower-case name of the app, which is the same as the "name" argument
  passed to this script

"""

import os
import subprocess
from argparse import ArgumentParser


#==============================================================================
# Constants
#    pathing and formatting mostly
#==============================================================================

# Constants
#------------------
#==== Root dirs
PWD   = os.environ["PWD"] # default installation dir
HOME  = os.environ["HOME"]
LOCAL = f'{HOME}/.local'
#==== Dest dirs
LOCAL_BIN  = f'{LOCAL}/bin'
LOCAL_APPS = f'{LOCAL}/share/applications'
DEFAULT_ARGS = '--file-download-options \'{"saveAs": true}\' -m'
#==== User-specific paths
APP_PATH  = f'{HOME}/.Apps/natified'
ICON_PATH = f'{HOME}/Chest/Resources/Icons/{{name}}.png'

# Desktop entry format
DESKTOP_ENTRY = f"""
[Desktop Entry]
Name={{title}}
Comment=Nativefied {{name}}
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
    help='full-path to installation directory, (default="~/.apps")')

adg('-a', '--auth', type=str, nargs=2,
    help=('<auth user-name> <auth user-password>; ',
        'WARNING: cleartext password will be logged by your shell history'))

adg('-u', '--internal-urls', type=int,
    help="apparently regex? I used int before and that worked") # nativefier ex: --internal-urls ".*?\.google\.*?" (but I just use num?)

adg('-t', '--tray', action='store_true',
    help='whether the app remains in tray when closed')

adg('-s', '--single-instance', action='store_false',
    help='only allow single instance of app')

adg('-c', '--counter', action='store_true',
    help='X number attached to window label for apps that support count, such as gmail')

# Parse user args
#----------------
args = vars(P.parse_args())


#==============================================================================
# Nativefy -
#    build nativefied app, app desktop entry, and symlink binary
#==============================================================================

# Nativefy
#---------------
def nativefy(opts):
    """ function that calls nativefier
    1. parse user args
    2. build nativefier command
    3. call nativefier
    4. make app binaries executable
    5. symlink binaries to local bin
    6. make desktop entry

    Params
    ------
    opts : dict
        the parsed args, contains all relevant nativefier flags
    """
    # Target url
    url  = opts['url']
    name = opts['name']

    # dest dir
    dest = opts['path']
    if not os.path.exists(dest): os.makedirs(dest)

    # icon
    icon_path = ICON_PATH.format(name=name)
    if os.path.exists(icon_path):
        opts['icon'] = icon_path

    # check authentication arg
    if opts['auth'] is not None:
        auth_usr, auth_pw = opts['auth']
        opts['basic-auth-username'] = auth_usr
        opts['basic-auth-password'] = auth_pw

    # Remove non-nativefier args
    del opts['url']
    del opts['path']
    del opts['auth']

    # Build nativefier command
    nativefier_cmd = "nativefier "
    for k, v in opts.items():
        if v:
            nativefier_cmd += f' --{k} {v}'
    nativefier_cmd += f' {url} {dest}'

    # Call nativefier
    # ---------------
    subprocess.run(nativefier_cmd, shell=True)
    print('\nFinished nativefier command')

    # Link binaries
    # -------------
    app_file_path = f'{dest}/{name}-linux-x64'
    bin_file_path = f'{app_file_path}/{name}'
    make_binary_exec(bin_file_path)
    symlink_binary(bin_file_path)

    # Make desktop entry
    make_desktop_entry(name, app_file_path)

    print('Finished nativefication process')


def make_desktop_entry(name, app_path):
    app_title = name.title()
    file_name = f'{name}.desktop'
    de = DESKTOP_ENTRY.format(title=app_title, name=name, path=app_path)
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
nativefy(args)
