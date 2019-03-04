troubleshooting
###############


Docker
******

Docker connection
=================

``docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?.``

``Failed to restart docker.service: Unit docker.service is masked``
`Solution <https://stackoverflow.com/a/53299880>`_

.. code-block:: bash

    sudo systemctl unmask docker.service
    sudo systemctl unmask docker.socket
    sudo systemctl start docker.service

-----

SSH
***

Remote sessions freezing
========================
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


PLACEHOLDER
