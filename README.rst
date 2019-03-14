=====
Chest
=====
A place I keep stuff for sync across machines. There is some organization, but I frequently update things and I don't worry about incomplete or redundant stuff or garbage commits. Cheat, Docs, and Reference all have somewhat similar content.

.. contents::

Cheat
=====
Files in cheat are partly cheatsheet, partly snippets. Plan is to eventually fork `cheat <https://github.com/cheat/cheat>`_, updating pathing/resolution logic to handle subdirs and add fuzzy searching for those fat cheats like ``Cheat/py``.


Docs
====
Docs currently has some notes on troubleshooting and command reference, mostly copied stuff from SO. This dir was separated from reference with the intention of housing a static/locally-hosted sphinx doc page (RTD), but I never build the site so.... probably going to consolidate this into Reference.

Dots
====
You guessed it. Chest was originally just dotfiles, but I kept adding stuff. The filenames are self-explanatory; my shell config file sources the primary dots.

Reference
=========
Messiest, most frequently updated dir in my chest. Stuff in Reference gets added, deleted, updated, relocated, consolidated all the time. Most of the reference content has been distilled into Cheats and Docs/howdoi.rst, and the stuff that remains is either WIP or not as cheatable.

Resources
=========
Assets and templates.

Scripts
=======
Orhpaned and cloned scripts. The good ones generally evolve into their own separate repos or reside in another VC'd scripts project. Only Scripts/gethub.py is used atm.

Setup
=====
After Dots, the most important thing in my chest. Contains my notes on my post-install setup process. Also contains a log of python packages I've used for specific purposes and others I found cool.

----

**License**:
Except where noted, original written content of the repo is licensed under `CC-BY-NC-ND-4.0`, while any software/code is licensed as `LGPL-3.0`. See LICENSE_.

.. Substitutions:


.. PROJECT FILES:


.. LOCAL FILES:
.. _LICENSE: LICENSE

.. EXTERNAL:
.. _pyenv: https://github.com/pyenv/pyenv
.. |pyenv| replace:: pyenv
