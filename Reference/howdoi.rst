############
How Do I....
############
do that 'thing'...again?
########################


**********
Conversion
**********

===
PDF
===
| **tl;dr :**
| 1. **convert to html**
| 2. **then to pdf** (through wkhtmltopdf)
| (somehow things get messed up in the intermediate latex conversion step)


- **convert .ipynb to pdf?**::

    # 2-step: nbconvert & wkhtmltopdf
    # -------------------------------
    jupyter nbconvert --to html my_notebook.ipynb
    wktmltopdf my_notebook.html my_notebook.pdf

    # 1-step: nbconvert with custom latex
    #    template through nb_pdf_template
    # -----------------------------------
    jupyter nbconvert --to pdf example.ipynb --template classicm
    # (to install nb_pdf_template):
    'pip install nb_pdf_template; python -m nb_pdf_template.install'

- **convert .rst to pdf?** (Still trying to find a good solution)::

    # Using docutils' rst2
    # --------------------
    rst2html README.rst > README.html
    wkhtmltopdf README.html README.pdf

======
Images
======

- **convert svg to png?** : ::

    inkscape -z -e test.png -w 1024 -h 1024 test.svg



#-----------------------------------------------------------------------------
#   Installation
#-----------------------------------------------------------------------------

>? Install py package from source?
#=======================
python setup.py install --prefix=$HOME/.local/bin


>? Install apt package without recommended|suggested?
#=======================
sudo apt --no-install-recommends --no-install-suggests install MY_PACKAGE



#-----------------------------------------------------------------------------
#   REMOTE, ssh etc
#-----------------------------------------------------------------------------

>? How to mount a remote dir to local machine?
#=======================
#----> sshfs
sshfs name@server:/path/to/folder /path/to/mount/point
OR
sshfs -o reconnect name@server:/path/to/folder /path/to/mount/point  # auto reconnect
OR
sshfs -o ssh_command='ssh -p <customport>' name@server:/path/to/folder /path/to/mount/point


>? Generate SSH key?
#=======================
mkdir ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t rsa


>? Send my client SSH key to server?
#=======================
ssh-copy-id <username>@<host>



#-----------------------------------------------------------------------------
# Chrome / web / extensions
#-----------------------------------------------------------------------------

>? How to see all installed extensions in Chrome?
#=======================
chrome://system



#-----------------------------------------------------------------------------
# Sessions / services
#-----------------------------------------------------------------------------

>? Save ipython session history|log?
#=======================
#-----> for current session
%history -f history.py

#-----> for all sessions:
%history -g -f full_history.py



#-----------------------------------------------------------------------------
# System packages
#-----------------------------------------------------------------------------

>? Check what dependees a package has?
#========================
apt-cache rdepends packagename


>? Apt install packages from text file?
#========================
cat pkg_list.txt | xargs sudo apt install


>? Remove|Uninstall list of files|packages from STDIN|txt?
#========================
#-----> For packages:
cat pkg_list.txt | xargs sudo apt remove --purge -y
EG:
sudo deborphan | xargs sudo apt remove --purge -y  # to remove all orphaned dependencies

#-----> For files:
cat stuff_i_dont_want.txt | xargs rm -rf -y



#-----------------------------------------------------------------------------
# System
#-----------------------------------------------------------------------------

>? Check my public IP?
#======================
inxi -i
OR
wget -O - -q icanhazip.com


>? Disable the insert key?
#=========================
# Step 1) what key is mapped to insert?
xmodmap -pke | grep -i insert

# Step 2) map ins key to null in ~/.Xmodmap
echo "keycode 90 =" >> ~/.Xmodmap

