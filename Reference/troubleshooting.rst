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
