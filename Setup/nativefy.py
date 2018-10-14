import os
import getpass
import shutil
import sys
from argparse import ArgumentParser

username = getpass.getuser() # returns $USER

APPSDIR = '.apps'
APPPATH = f'/home/{username}/{APPSDIR}/'

#nativefier "https://keep.google.com" -n google-keep -a x64 -p linux -i icons/google-keep-icon.png
# nativefy https://keep.google.com google-keep

P = ArgumentParser()
adg = P.add_argument
adg('url', type=str)
adg('-n','--name', type=str, required=True)
adg('-i', '--icon', type=str)
adg('-t', '--tray', function='store_true')
adg('--single', function='store_true')


DEFAULT_ARGS = '-a x64 -p linux --honest -m'
#--honest -m -i ./Icons/HackerRank_logo.png
#

# nativefier "https://www.hackerrank.com/dashboard" -n hacker-rank -a x64 -p linux --title-bar-style hidden --honest -m -i ./Icons/HackerRank_logo.png


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
getapp()
