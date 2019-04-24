============
How Do I....
============

.. contents::



---
PDF
---


Conversion
==========
Conversion to pdf often involves an intermediate step where the document is converted to latex before pdf. This often messes things up.

I've found it's easier to first convert to html, then pdf.

If you can find one, a latex or css template for converting from some format to pdf is generally the best.


Notebook to pdf
---------------
For *any* solution, you will have to make some compromise.

The trouble with notebook to pdf conversion is that you cannot get clean single pages. Some methods will leave 90% of a page blank so that a cell or something does not get truncated.

So far, I've only managed to work around the paging issue by manually formatting cells by guestimation.

A cursory search did not yield any solutions for embedded styling directives or something for pagebreaks, but there may be more there.

**Methods used**:

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
----------
(Still trying to find a good solution). Unlike the misadventures in notebook conversion, I think there may be more resources for rst to pdf I have not found yet.

.. code-block:: bash

    # Using docutils' rst2
    # --------------------
    rst2html README.rst > README.html
    wkhtmltopdf README.html README.pdf


Manipulation
============
There are LOADS of CLI tools for manipulating and modifying pdfs.

If these solutions do not work for you, just google whatever you need to do.

Crop pdf
--------
I found the top hits on SO and such to be very tedious.
They all tend to use library modules packaged with poppler or texlive.

The issue has been that the defaults are generally too aggressive in cropping.
You can specify margins, but even still, they often crop sparse pages
to an entirely different size than normal pages.


**Here's the best way**, using a interfacing script::

    # Install pdf pkgs (texlive gives you pdfcrop)
    sudo apt intall --no-install-recommends --no-install-suggests texlive-extra-utils

    # Use python pkg interface
    pip install -U pdfCropMargins
    pdf-crop-margins -s -u paper.pdf



**hard way**:

To crop with all pages at consistent page size: https://tex.stackexchange.com/questions/166758/how-do-i-make-pdfcrop-output-all-pages-of-the-same-size

1. ``pdfcrop --verbose myfile.pdf cropfile.pdf > crop.log``
2. Open ``crop.log``, get all  lines with ``%%HiResBoundingBox:``, and strip those lines so its just the space separated nums on the lines
3. open that log in python, and get bbox as follows

.. code-block:: python

    import pyperclip
    with open('crop.log') as log:
        rlines = [line.split('\n') for line in log.read.strip().split('\n')]

        a,b,c,d = 0,0,0,0
        for w,x,y,z in rlines:
            a = max(a, eval(w))
            b = max(b, eval(x))
            c = max(c, eval(y))
            d = max(d, eval(z))
        pyperclip.copy(f'pdfcrop --bbox "{a} {b} {c} {d}"')

4. ``pdfcrop --box "<the nums>" myfile.pdf cropfile.pdf``


remove a watermark
------------------

.. code-block:: bash

    #=== cut watermark text from pdf code
    sed -e "s/watermarktextstring/ /g" <input.pdf >unwatermarked.pdf
    #=== fix modified pdf
    pdftk unwatermarked.pdf output fixed.pdf && mv fixed.pdf unwatermarked.pdf

extract a range of pages
------------------------
NB: pdfjam is part of the texlive package.

.. code-block:: bash

    pdfjam <input file> <page ranges> -o <output file>
    # eg:
    pdfjam original.pdf 3-8 -o out.pdf
    pdfjam original.pdf 3-8,15-29,63-69 -o out.pdf

extract pdf pages as png
------------------------
Check out: https://askubuntu.com/questions/50170/how-to-convert-pdf-to-image

.. code-block:: bash

    # output each page in PDF, with name format `outputname-01.png`
    pdftoppm input.pdf outputname -png

    # Single page
    pdftoppm input.pdf outputname -png -f pgnum -singlefile

    # The default resolution, 150 dpi, is kind of shit, so
    # you can try increasing resolution to RES dpi like:
    pdftoppm input.pdf outputname -png -rx RES -ry RES

----


------
Images
------

Conversion
==========

**convert svg to png**:

    ``inkscape -z -e test.png -w 1024 -h 1024 test.svg``

**convert to monochrome**:

    ``convert input_image.png -monochrome output.png``

    Some other options, depending on the result::

        # higher resolution
        convert input_image.png -density 150 output.png

        # dithering
        convert input_image.png -remap pattern:gray50 output.png


----

----
Keys
----

SSH
===

**generate ssh key**:

.. code-block:: bash

    ssh-keygen -t rsa -b 4096 -C "my_email@abc.com"
    # just accept defaults

**add SSH key to ssh-agent**:

.. code-block:: bash

    eval "$(ssh-agent -s)"
    # Should see print of agent PID
    ssh-add ~/.ssh/id_rsa


**add my SSH key to server**:

.. code-block:: bash

    #=== add to server (from local)
    ssh-copy-id <username>@<host>

    #=== copy ssh pubkey
    cat ~/.ssh/id_rsa.pub | xclip -selection clipboard


GPG
===

**generate gpg key**:

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

-----

------------
Installation
------------

Python
======

**Install py package from source**:

.. code-block:: bash

    python setup.py install --prefix=$HOME/.local/bin


Apt
===

**Install package without recommended|suggested**::

    sudo apt --no-install-recommends --no-install-suggests install MY_PACKAGE


------

-------------
Remote Server
-------------


**mount remote dir to local**::

    # basic connection
    sshfs name@server:/path/to/folder /path/to/mount/point

    # Auto reconnect if drop
    sshfs -o reconnect name@server:/path/to/folder /path/to/mount/point

    # Custom port
    sshfs -o ssh_command='ssh -p <customport>' name@server:/path/to/folder /path/to/mount/point


**Send my client SSH key to server**::

    ssh-copy-id <username>@<host>


-----

------
Python
------

ipython
=======

**Save ipython session history|log**:

.. code-block:: python

    #-----> for current session
    %history -f history.py

    #-----> for all sessions:
    %history -g -f full_history.py

----

---------------
Troubleshooting
---------------

Ubuntu/Linux
============

Slow boot
---------
This has been a persistent problem for **all** my machines with xubuntu 18.04. None had slow-boot issues with 16.04.

After hours of googling and trying out a bunch of stuff (including a disastrous modification to lightdm/wayland that was only meant for ubuntu and not xubuntu), **I still have not found a solution.**

This is probably the only issue I've ever had where I have not found a solution online, and there doesn't seem to be much discussion, despite it's **consistent** behavior across different machines and hardware.

I had a boot time < 4s on 16.04. With 18.04, boot-times are consistently around 15~20s.

**HOW TO REDUCE BOOT TIME**:

1. See what processes are taking the longest:

.. code-block:: bash

    systemd-analyze blame
    systemd-analyze critical-chain
    systemd-analyze time


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

4. ``NetworkManager-wait-online.service`` is another  usual suspect. You can just disable it::

    sudo systemctl disable NetworkManager-wait-online.service


Black screen on boot
--------------------
The primary issue is a **hanging black screen** on boot. This phenomenon is apparently **NOT** logged by any of the typical system processes--eg ``systemd-analyze`` won't register this boot lag for any process.

The system boots, normally then hangs on a blank, black screen for approximately 15~20s, and it seems like it can persist longer *if* you do not spam the keyboard (which seems to interrupt it).

**WHAT I'VE TRIED**:

- ANYTHING involving grub2. Yes, really. Everything
- doing something with lightdm and wayland, as suggested by https://askubuntu.com/a/1053697. This literally broke my system, and took me all day to recover. Turns out xubuntu doesnt use gdm3 or wayland or whatever.
- Tinkering with nouveau, nvidia, mesa stuff


Fonts
=====
This is a nightmare on linux.

Check your dpi::

    xdpyinfo | grep resolution

    # dpi plus res
    xdpyinfo | grep -B2 resolution

-----

-------------
Miscellaneous
-------------

**Check my public ip**::

    inxi -i
    # or
    wget -O - -q icanhazip.com


**Disable the ins key**

.. code-block:: bash

    # Figure out what is mapped to insert key
    xmodmap -pke | grep -i insert

    # Map ins key to null in ~/.Xmodmap
    echo "keycode 90 =" >> ~/.Xmodmap


**Prevent tor from starting automatically**::

    sudo systemctl disable tor.service



Chrome & Browser
================

- See all installed extensions: navigate to ``chrome://system``