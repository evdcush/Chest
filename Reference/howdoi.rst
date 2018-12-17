############
How Do I....
############
do that 'thing'...again?
########################


.. contents:: Table of Contents
.. section-numbering::


PDF
===

conversion
----------

| **For most .thing --> .pdf :**
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


Manipulation
------------

- **crop PDF?**::
    
    sudo apt install --no-install-recommends --no-install-suggests texlive-extra-utils
    pdfcrop my_doc.pdf cropped_my_doc.pdf

- **remove a watermark?**::
    
    #=== cut watermark text from pdf code
    sed -e "s/watermarktextstring/ /g" <input.pdf >unwatermarked.pdf
    #=== fix modified pdf
    pdftk unwatermarked.pdf output fixed.pdf && mv fixed.pdf unwatermarked.pdf



Images
======

- **convert svg to png?**::

    inkscape -z -e test.png -w 1024 -h 1024 test.svg


Keys
====

SSH
---

- **generate ssh key?**::

    ssh-keygen -t rsa -b 4096 -C "my_email@abc.com"
    # just accept defaults

- **add SSH key to ssh-agent?**::
    
    eval "$(ssh-agent -s)"
    # Should see print of agent PID
    ssh-add ~/.ssh/id_rsa

- **add my SSH key to...?**::

    #=== add to server (from local)
    ssh-copy-id <username>@<host>

    #=== copy ssh pubkey
    cat ~/.ssh/id_rsa.pub | xclip -selection clipboard


GPG
---

- **generate gpg key?**::
    
    #  Part of the process involves "generating enough 
    #  bits of entropy" for random seed, so best to first
    #  install some helpful utils for that
    sudo apt install rng-tools

    # Now go through gpg setup, selecting what you want
    gpg --full-generate-key

    # Now it may say to do stuff for entropy, try this:
    sudo rngd -r /dev/urandom

    # you should now have your key


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

