############
How Do I....
############

.. contents:: Table of Contents
.. section-numbering::


PDF
===

conversion
----------
Conversion to pdf often involves an intermediate step where the document is converted to latex before pdf. This often messes things up. 

I've found it's easier to first convert to html, then pdf.


``.ipynb`` to pdf
^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # 2-step: nbconvert & wkhtmltopdf
    # -------------------------------
    jupyter nbconvert --to html my_notebook.ipynb
    wktmltopdf my_notebook.html my_notebook.pdf

    # 1-step: nbconvert with custom latex
    #    template through nb_pdf_template
    # -----------------------------------
    jupyter nbconvert --to pdf example.ipynb --template classicm
    # (to install nb_pdf_template):
    pip install nb_pdf_template; python -m nb_pdf_template.install


rst to pdf
^^^^^^^^^^
(Still trying to find a good solution)

.. code-block:: bash

    # Using docutils' rst2
    # --------------------
    rst2html README.rst > README.html
    wkhtmltopdf README.html README.pdf


Manipulation
------------
There are LOADS of CLI tools for manipulating and modifying pdfs. Just google whatever you need to do.

crop PDF
^^^^^^^^
.. code-block:: bash

    sudo apt install --no-install-recommends --no-install-suggests texlive-extra-utils
    pdfcrop my_doc.pdf cropped_my_doc.pdf

remove a watermark
^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    #=== cut watermark text from pdf code
    sed -e "s/watermarktextstring/ /g" <input.pdf >unwatermarked.pdf
    #=== fix modified pdf
    pdftk unwatermarked.pdf output fixed.pdf && mv fixed.pdf unwatermarked.pdf

extract a range of pages
^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    pdfjam <input file> <page ranges> -o <output file>
    # eg:
    pdfjam original.pdf 3-8 -o out.pdf



Images
======

Conversion
----------

convert svg to png
^^^^^^^^^^^^^^^^^^
``inkscape -z -e test.png -w 1024 -h 1024 test.svg``


Keys
====

SSH
---

generate ssh key
^^^^^^^^^^^^^^^^
.. code-block:: bash

    ssh-keygen -t rsa -b 4096 -C "my_email@abc.com"
    # just accept defaults

add SSH key to ssh-agent
^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash
    
    eval "$(ssh-agent -s)"
    # Should see print of agent PID
    ssh-add ~/.ssh/id_rsa


add my SSH key to server
^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    #=== add to server (from local)
    ssh-copy-id <username>@<host>

    #=== copy ssh pubkey
    cat ~/.ssh/id_rsa.pub | xclip -selection clipboard


GPG
---

generate gpg key
^^^^^^^^^^^^^^^^
.. code-block:: bash
    
    #  Part of the process involves "generating enough 
    #  bits of entropy" for random seed, so best to first
    #  install some helpful utils for that
    sudo apt install rng-tools

    # Now go through gpg setup, selecting what you want
    gpg --full-generate-key

    # Now it may say to do stuff for entropy, try this:
    sudo rngd -r /dev/urandom

    # you should now have your key


Installation
============

Python
------

Install py package from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    python setup.py install --prefix=$HOME/.local/bin


Apt
---

Install package without recommended|suggested
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    sudo apt --no-install-recommends --no-install-suggests install MY_PACKAGE



Remote Server
=============

via ssh
-------

mount remote dir to local
^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash
    
    # basic connection
    sshfs name@server:/path/to/folder /path/to/mount/point

    # Auto reconnect if drop
    sshfs -o reconnect name@server:/path/to/folder /path/to/mount/point

    # Custom port
    sshfs -o ssh_command='ssh -p <customport>' name@server:/path/to/folder /path/to/mount/point


Send my client SSH key to server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash
    
    ssh-copy-id <username>@<host>



Python
======

ipython
-------

Save ipython session history|log
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    #-----> for current session
    %history -f history.py

    #-----> for all sessions:
    %history -g -f full_history.py



Packages
========

Utils
-----

Check what package dependees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    apt-cache rdepends packagename


Adding and Removing
-------------------

Apt install packages from text file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    cat pkg_list.txt | xargs sudo apt install


remove list of files or packages from STDIN or txt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    #-----> For packages:
    cat pkg_list.txt | xargs sudo apt remove --purge -y
    EG:
    sudo deborphan | xargs sudo apt remove --purge -y  # to remove all orphaned dependencies

    #-----> For files:
    cat stuff_i_dont_want.txt | xargs rm -rf -y

Troubleshooting
===============

Ubuntu/Linux
------------

Slow boot
^^^^^^^^^
This has been a persistent problem for **all** my machines with xubuntu 18.04. None had slow-boot issues with 16.04.

After hours of googling and trying out a bunch of stuff (including a disastrous modification to lightdm/wayland that was only meant for ubuntu and not xubuntu), **I still have not found a solution.** 

This is probably the only issue I've ever had where I have not found a solution online, and there doesn't seem to be much discussion, despite it's **consistent** behavior across different machines and hardware.

I had a boot time < 4s on 16.04. With 18.04, boot-times are consistently around 15~20s.

**HOW TO REDUCE BOOT TIME**:

1. See what processes are taking the longest:

.. code-block:: bash

    systemd-analyze blame

2. Find the slowest processes, and disable them or modify their start processes. If there is a specific thing taking significantly longer than other processes, it's best to google that process to see how other users handled it first.


3. ``apt-daily.service``. This is a known bug with 18.04; this process is not supposed to run during boot. The "workaround" involves editing the timer via ``sudo systemctl edit apt-daily.timer``, but this only worked temporarily, I'm not sure why. I was able to get a persistent fix by instead directly editing the timer file:


.. code-block:: bash

    # first backup
    sudo cp /lib/systemd/system/apt-daily{,.bkp}.timer

    # now replace the following [Timer] settings
    sudo vi /lib/systemd/system/apt-daily.timer
    [Timer]
    OnBootSec=15min
    OnUnitActiveSec=1d
    AccuracySec=1h
    RandomizedDelaySec=30min



Black screen on boot
""""""""""""""""""""
The primary issue is a **hanging black screen** on boot. This phenomenon is apparently **NOT** logged by any of the typical system processes--eg ``systemd-analyze`` won't register this boot lag for any process.

The system boots, normally then hangs on a blank, black screen for approximately 15~20s, and it seems like it can persist longer *if* you do not spam the keyboard (which seems to interrupt it).

**WHAT I'VE TRIED**:

- ANYTHING involving grub2. Yes, really. Everything
- doing something with lightdm and wayland, as suggested by https://askubuntu.com/a/1053697. This literally broke my system, and took me all day to recover. Turns out xubuntu doesnt use gdm3 or wayland or whatever.
- Tinkering with nouveau, nvidia, mesa stuff




Unsorted
========

Check my public ip
------------------
.. code-block:: bash

    inxi -i
    # or
    wget -O - -q icanhazip.com


Disable the ins key
-------------------
1. figure out what key is mapped to insert

.. code-block:: bash
    
    xmodmap -pke | grep -i insert

2. map ins key to null in ~/.Xmodmap

.. code-block:: bash

    echo "keycode 90 =" >> ~/.Xmodmap


Chrome & Browser
----------------

- See all installed extensions: navigate to ``chrome://system``



