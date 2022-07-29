===============
troubleshooting
===============


Applications
============

Okular
------

Icons are missing
^^^^^^^^^^^^^^^^^
.. code-block::

    # Install
    sudo apt install qt5ct breeze-icon-theme

    # Run:
    qt5ct --platformtheme qt5ct
    # Set the icon-theme to breeze

    # Set the env variable:
    sudo vi /etc/environment
    # Add this line:
    QT_QPA_PLATFORMTHEME="qt5ct"

    # Copy desktop file:
    cp /usr/share/applications/org.kde.okular.desktop ~/.local/share/applications

    # Modify the .desktop file by changing the line:
    Exec=okular %U
    # to
    Exec=okular --platformtheme qt5ct %U





Audio
=====

Bluetooth headset connects but not available as output device
-------------------------------------------------------------
I tried a few things, but since I didn't A/B test the solutions, I'm not certain which thing fixed it.
But I believe it was this: https://askubuntu.com/a/1243890/601020

> This is a bug (pulseaudio #832, launchpad #1866194) with new version of pulseaudio in Ubuntu 20.04 where old config values have not been cleared and it is not routing to the correct device. The solution is to delete the old config and restart pulseaudio::

    mv ~/.config/pulse/ ~/.config/pulse.old
    systemctl --user restart pulseaudio

    # Another solution based on this exec'd this after the mv ~/.config/pulse:
    # https://superuser.com/a/1623869/865546
    pulseaudio --k && pulseaudio --start


You may also need to restart.
You may also need to remove the bluetooth device and re-pair it.

At the same time, I also installed blueman or bluez. But I uninstalled it and I don't think that was the fix.


-------


Startup etc
===========

disable autostart
-----------------

To disable teamviewer from autostart::

    sudo teamviewer --daemon disable

------

CIFS mount
==========

cifs mount using env vars for usr and pwd
-----------------------------------------
This is another one of those instances where you want to consistently
call some command in shell that uses sensitive information (such as a
password).

There are many potential solutions, here is one from SO using a
credentials file: (https://askubuntu.com/a/67418/601020)::

    # Create a text file, eg $HOME/.Cifs.cred to store the credentials:
    user=your-username-on-cifs-server
    password=the-password
    domain=leave-this-blank-unless-really-using-windows-domain

    # Protect the $HOME/.Cifs.cred file; run this command
    # NB: this changes perm to -rw------ and is owned by user
    chmod go-rw $HOME/.Cifs.cred

    # Now mount CIFS share to target directory using the creds file:
    sudo mount -t cifs -o cred=$HOME/.Cifs.cred,uid=`id -u`,gid=`id -g`, //server/data /mnt/share_data





deb stuff
=========

``dpkg: warning: files list file for package 'X' missing``:

    sudo apt remove X
    sudo apt autoclean




Docker
======

Docker connection
-----------------

``docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?.``

``Failed to restart docker.service: Unit docker.service is masked``
`Solution <https://stackoverflow.com/a/53299880>`_

.. code-block:: bash

    sudo systemctl unmask docker.service
    sudo systemctl unmask docker.socket
    sudo systemctl start docker.service

-----

Fonts
=====
Look like garbage on linux.

**Consult:** https://pandasauce.org/post/linux-fonts/


Chromium
--------
Consulting the afforementioned pandasauce post, the author, Georgi Boiko, has a gist for manually patching binary releases of chrome to fix font (force full hinting and subpixel positioning).

**The gist:** https://gist.github.com/pandasauce/398c080f9054f05bee6e1c465416b53b

He uses a tool called `radiff2 <https://r2wiki.readthedocs.io/en/latest/tools/radiff2/>`_, which is part of the ``radare2`` package. radiff shows difference between two binaries.

So to patch chromium like in that gist::

    # Install deps and parent lib
    sudo apt install xdot radare2


-----


NPM
===
Setup or usage issues


proxy
-----
Explicitly set the HTTP and HTTPS proxy (apparently npm not read these vars from env)::

    # http proxy
    npm config set proxy http://proxy.example.com:8080

    # https proxy
    npm config set https-proxy http://proxy.example.com:8080


-----

ROS
===
ROS is still on python2, so you'll likely have issues with PYTHONPATH and
annoying coupling issues between your typical venv and system-site packages.

ModuleNotFoundError: No module named 'deez-nuts'
------------------------------------------------
So you've gotten this error for: ``apt-pkg``, ``rospkg``, ``defusedxml``.

**First step:** make sure you have these packages installed
- I installed to both system, and venv:

    ``sudo apt install -y python-apt python3-apt python-rospkg python-defusedxml python3-defusedxml``
    ``pip install rospkg defusedxml``

BUT this didnt fix anything. Realizing some python2 stuff, I tried adjusting
the system default python::

    sudo update-alternatives python
    # then select python2

**THIS FIXED IT**


-----

SSH
===

Remote sessions freezing
------------------------
You need to properly configure the ssh config files on both server and client. This solution from an answer on SO: `"How can I keep my SSH sessions from freezing?" <https://unix.stackexchange.com/a/200256>`_

**On the client-side ssh config:**

.. code-block:: bash

    sudo vi /etc/ssh/ssh_config
    # (in ssh_config)
    Host *
    ServerAliveInterval 100

With ``ServerAliveInterval 100``, the client will send a null packet to the server every 100 seconds to keep the connection alive


**On the server-side sshd config:**

.. code-block:: bash

    sudo vi /etc/ssh/sshd_config
    # Add/edit the following lines:
    ClientAliveInterval 60
    TCPKeepAlive yes
    ClientAliveCountMax 10000


With ``ClientAliveInterval 60``, the server will wait 60s before sending a null packet to the client to keep the connection alive.

With ``ClientAliveCountMax``, the server will send alive messages to the client even though it has not received any message back from the client.

**Finally, restart the ssh service:** ``sudo systemctl restart sshd.service``


------


Time & Date Stuff
=================
Your clock is incorrect, and you are trying to fix it or synchronize.

Typically, you just need to install ntp: `sudo apt install ntp`.

But chances are, you don't have such a simple case. You'll also see people
recommending to `sudo ntpdate ntp.ubuntu.com`.

What they should actually say is::

    sudo service ntp stop  # since "socket" is in use
    sudo ntpdate ntp.ubuntu.com
    sudo service ntp start


But, if you're on a suffocating company proxy that blocks everyhting,
none of the above will work, since the `123` port will be blocked.

So, just manual fix::

    sudo date -s "$(wget -qSO- --max-redirect=0 google.com 2>&1 | grep Date: | cut -d' ' -f5-8)Z"



**Reference**:

- https://askubuntu.com/questions/429306/ntpdate-no-server-suitable-for-synchronization-found
- https://superuser.com/questions/639202/updating-time-ntpdate3108-the-ntp-socket-is-in-use-exiting
- https://askubuntu.com/questions/201133/can-i-use-ntp-service-through-a-proxy


------


Xorg & Display issues
=====================

**How to restart xorg?**

.. code-block:: bash

    sudo systemctl restart display-manager

    # find out which display manager your ubuntu has (not actually relevant)
    cat /etc/X11/default-display-manager


**Display not loaded on GPU? Resolution is fixed to very low setting?**

.. code-block:: bash

    sudo /etc/init.d/lightdm restart
