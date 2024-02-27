###################
CUDA Happy Fun Time
###################

Installation, updating, troubleshooting, and other fun and painfree good times
with Nvidia & CUDA on Linux.

Troubleshooting
###############


``dpkg-error``::

    Preparing to unpack .../libnvidia-gl-550_550.54.14-0ubuntu1_amd64.deb ...
    dpkg-query: no packages found matching libnvidia-gl-535
    Unpacking libnvidia-gl-550:amd64 (550.54.14-0ubuntu1) ...
    dpkg: error processing archive /var/cache/apt/archives/libnvidia-gl-550_550.54.14-0ubuntu1_amd64.deb (--unpack):
     trying to overwrite '/usr/lib/x86_64-linux-gnu/libnvidia-api.so.1', which is also in package libnvidia-extra-545:amd64 545.23.08-0ubuntu1
    dpkg-deb: error: paste subprocess was killed by signal (Broken pipe)
    Errors were encountered while processing:
     /var/cache/apt/archives/libnvidia-gl-550_550.54.14-0ubuntu1_amd64.deb
    E: Sub-process /usr/bin/dpkg returned an error code (1)

**Fix**:
https://askubuntu.com/questions/1062171/dpkg-deb-error-paste-subprocess-was-killed-by-signal-broken-pipe

.. code-block::

    sudo dpkg -i --force-overwrite /var/cache/apt/archives/libnvidia-gl-550_550.54.14-0ubuntu1_amd64.deb
