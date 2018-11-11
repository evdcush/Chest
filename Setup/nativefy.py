import os
import subprocess
from argparse import ArgumentParser


#==============================================================================
# Constants & Utils -
#    pathing and formatting mostly
#==============================================================================

# Utility classes
#------------------
class AttrDict(dict):
    # just a dict mutated/accessed by attribute instead index
    # NB: not pickleable
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Constants
#------------------
PWD   = os.environ["PWD"] # default installation dir
HOME  = os.environ["HOME"]
LOCAL = f'{HOME}/.local'
LOCAL_BIN  = f'{LOCAL}/bin'
LOCAL_APPS = f'{LOCAL}/share/applications'
DEFAULT_ARGS = '--file-download-options \'{"saveAs": true}\' -m'

DESKTOP_ENTRY = f"""
[Desktop Entry]
Name={{title}}
Comment=Nativefied {{name}}
Exec={LOCAL_BIN}/{{name}}
Terminal=false
Type=Application
Icon={{path}}/{{name}}/resources/app/icon.png
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

adg('-p', '--path', type=str, default=PWD,
    help='full-path to installation directory, (default = pwd)')

adg('-a', '--auth', type=str, nargs=2,
    help=('<auth user-name> <auth user-password>; ',
        'WARNING: cleartext password will be logged by your shell history'))

adg('-u', '--user-agent', type=str,
    help='user-agent; defaults to chrome-browser')

adg('--internal-urls', type=int,
    help="apparently regex? I used int before and that worked") # nativefier ex: --internal-urls ".*?\.google\.*?" (but I just use num?)

adg('-t', '--tray', function='store_true',
    help='whether the app remains in tray when closed')

adg('--single', function='store_true',
    help='only allow single instance of app')

adg('--counter', function='store_true',
    help='X number attached to window label for apps that support count, such as gmail')

# Parse user args
#----------------
args = AttrDict(P.parse_args())


#==============================================================================
# Nativefy -
#    build nativefied app, app desktop entry, and symlink binary
#==============================================================================

# Nativefy
#---------------
def nativefy(opts):
    """ function that calls nativefier
    - cd to desired app path if different from current
    - call nativefier

    Params
    ------
    opts : AttrDict
        the parsed args, contains all relevant nativefier flags
    """
    pass


def make_desktop_entry(name, app_path):
    app_title = name.title()
    file_name = f'{name}.desktop'
    de = DESKTOP_ENTRY.format(title=app_title, name=name, path=app_path)
    with open(f'{LOCAL_APPS}/{file_name}', 'w') as entry:
        entry.write(de)
        print(f'\n{file_name} written to {LOCAL_APPS}')


def symlink_binary(from_path, to_path=LOCAL_BIN):
    subprocess.run(f'ln -s {from_path} {to_path}', shell=True)
    print(f'\nsymlinked binary -\nFROM: {from_path}\n  TO: {to_path}')


def make_binary_exec(bin_file_path):
    subprocess.run(f'chmod +x {bin_file_path}', shell=True)
    print('\nMade binary executable')













# REFERNCE GIST
def getapp():
    os.system('clear')

    # App info from user input
    #-------------------------------
    #appurl = str(input("App url: ").lower())
    ##==== verify appurl
    #if not appurl.split('//')[0] in ['http:', 'https:']:
    #    appurl = 'https://' + appurl
    #appname = str(input("App name: "))




    # Pathing
    #-----------------------
    # THIS WILL ALL BE DONE BY PARSER
    #apppath=f'{appname.lower()+'/'}
    #appsh =
    appsh='/home/'+username+'/webapps/'+appname+'.sh'
    appshort='/home/'+username+'/webapps/'+appname+'.desktop'
    alias="alias NewApp='python3 /home/"+username+"/webapps/NewApp.py'"
    newapp=os.path.isfile('/home/' + username + '/webapps/NewApp.py')

    os.chdir('/home/'+username)
    os.system('nativefier '+appurl+' --name '+appname.lower())

    shutil.move('/home/'+username+'/'+appname.lower()+'-linux-x64', apppath)

    os.system("echo '#!/usr/bin/env bash'>"+appsh)
    os.system('echo "' +apppath+appname.lower()+'">>'+appsh)
    os.system('chmod +x '+appsh)


    os.system('echo "[Desktop Entry]">'+appshort)
    os.system('echo "Version=1.0">>'+appshort)
    os.system('echo "Type=Application">>'+appshort)
    os.system('echo "Exec='+appsh+'">>'+appshort)
    os.system('echo "Name='+appname+'">>'+appshort)
    os.system('echo "Icon='+apppath+'/resources/app/icon.png">>'+appshort)
    os.system('chmod +x '+appshort)

    if newapp:
        os.remove('/home/' + username + '/webapps/NewApp.py')
        os.system(
            'wget https://github.com/Jafesu/Nativefier-Auto/raw/master/NewApp.py -P /home/' + username + '/webapps/')
    else:
        os.system(
            'wget https://github.com/Jafesu/Nativefier-Auto/raw/master/NewApp.py -P /home/' + username + '/webapps/')
    os.system('echo "'+alias+'">>~/.bash_aliases')
    os.system('source ~/.bash_aliases')
    os.system('source ~/.bashrc')
