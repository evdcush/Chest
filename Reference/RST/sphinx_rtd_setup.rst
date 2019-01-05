################################
Setting up sphinx with RTD theme
################################


Sphinx setup
============

1. Install sphinx and the rtd theme ::

    pip install -U docutils sphinx sphinx_rtd_theme

2. Make a directory for your documentation ::

    mkdir my_docs && cd my_docs


3. Generate sphinx docs with quickstart. This process will walk you through the setup for sphinx. I chose defaults for most questions ::

    sphinx-quickstart

Building sphinx
---------------
Your root documentation directory (the directory in which you called ``sphinx-quickstart``) should look something like this::

    $ tree 
    .
    ├── conf.py
    ├── index.rst
    ├── Makefile
    ├── _static
    └── _templates

In order to build the sphinx documentation, simply call ``make html`` in the root docs dir (``my_docs``)::

    make html

This will generate the ``_build`` directory, which has two subdirs: ``doctrees`` and ``html``.

Rendered docs
^^^^^^^^^^^^^
The rendered documentation is located at::

    my_docs/_build/html/index.html

And you can view it as a static html page by opening ``index.html`` in your web browser (eg ``chromium-browser index.html``).


Read-the-docs theme
-------------------
Located in the sphinx docs root directory is a file ``conf.py``.

In ``conf.py`` **modify or add** the following lines::

    html_theme = "sphinx_rtd_theme"
    html_theme_path = ["_themes",]

Now, when you ``make html``, your files will be using the RTD theme!



-----

Adding content
==============
WIP! Still figuring this out, but it looks fairly straight forward.


There are two steps to adding content to your documentation.

Step 1: rST file
----------------
Write your content in reStructuredText. For example, let's make a sample file, ``salamander.rst``, with the following content::

    ###########
    Salamanders
    ###########
    
    Salamanders look like wet lizards, but actually they are amphibians, grouped under the order *Urodela*.
    
    Cool Facts
    ==========
    Salamanders look pretty cool, but they also have some cool traits.
    
    - Salamanders are capable of regenerating lost limbs, and other damaged parts of their bodies
    - Some salamanders have skin that contains a powerful poison called *tetrodotoxin*
        - These poisonous dudes often have really cool warning colors too
    
    
    What about newts?
    =================
    Newts are a subfamily of salamanders, and are the most common type of salamander.



Step 2: index.rst
-----------------
The ``index.rst`` file is a special file for your sphinx documentation. It is the root node in the hierarchical structure of your documents, and all your documentation files must be specified here.

In the index file, you will see the following section ::

    .. toctree::
   :maxdepth: 2
   :caption: Contents:

Add your file, ``salamanders.rst`` to the index simply by adding it's name under the ``toctree`` directive::

    .. toctree::
   :maxdepth: 2
   :caption: Contents:
   salamanders


Now, when you call ``make html``, and check out your new ``index.html`` file, you should see your documentation has a page for salamanders!



Tips and Tricks
===============

| **add ``_build`` to your gitignore**:
| The ``_build`` directory contains the *rendered* documentation files. We only want the documentation source content to be in version control, so it's best to ignore the ``_build`` dir.
