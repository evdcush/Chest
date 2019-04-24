===============
troubleshooting
===============

Startup etc
===========

disable autostart
-----------------

To disable teamviewer from autostart::

    sudo teamviewer --daemon disable

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


Xorg
====

**How to restart xorg?**

.. code-block:: bash

    sudo systemctl restart display-manager

    # find out which display manager your ubuntu has (not actually relevant)
    cat /etc/X11/default-display-manager