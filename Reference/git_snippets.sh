#!/bin/bash

# NICE RESOURCES?
# https://github.com/wyntau/dotfiles


# push a local branch
#git push <remote-name> <local-branch-name>:<remote-branch-name>
git checkout -b <branch>
git push -u <remote> <branch>

# Create branch with no history/code
# ----------------------------------
git checkout --orphan my_branch

# to unstage automatically added code:
git rm --cached -r .

# to remove automatically added code:
git rm -rf .

# To pull all new remote branches from remote:
git remote update


# Undo diffs to file mode/permissions
# -----------------------------------
# NB: optional targeting of file mode undo, but generalyl fine i think
git diff -p -R --no-color . | \
grep -E "^(diff|(old|new) mode)" --color=never | \
git apply


# Prettier git log, with graphical branching
# ------------------------------------------
git log --graph --decorate --oneline --all


# ============== #
# DISABLE PAGING #
# ============== #
# To disable paging (for log, or ANY git command that uses a pager)
# simply append `--no-pager` immediately after `git` and before the git command.
git --no-pager log
git --no-pager diff
# etc...

# git diff files from different branches
# --------------------------------------
git diff branch_a branch_b -- my_file.py # can remove -- if compare work tree

# Undo last merge
# ---------------
git reset --hard ORIG_HEAD

# Change commit date
# ------------------
git commit --amend --date "Wed Jan 7 11:21:46 2019 -0800"


# -------------------------------------
# "Merge" single file from other_branch
# -------------------------------------
#==== Stage files that would be merged by git, without committing yet
#     choose which ones you want
git merge --no-ff --no-commit other_branch

# if you dont want one of the files from the above command:
git checkout HEAD file1

#==== Just want the version from other_branch (overwrites)
git checkout other_branch file1

#========================================


# Keep a file in git, but do not track history
# --------------------------------------------
# MORE TROUBLE THAN IT WORTH, DONT
#git update-index --skip-worktree <file_name>
#git update-index --no-skip-worktree <file_name> # opposite
# it will read as up to date, and changes will not be flagged as changes

#==== to keep a file in a git repo, but will not update and do not want updates:
# git update-index --assume-unchanged <file_name>
# git update-index --no-assume-unchanged <file_name> # opposite

#==== How to list files ignored via --skip-worktree
git ls-files -v . | grep ^S

# List files modified by current branch.
git whatchanged --name-only --pretty="" origin..HEAD
# (with dups removed):
git whatchanged --name-only --pretty="" origin..HEAD | sort -u


# See what files are tracked in git history
# (even deleted files)
# -----------------------------------------
git log --pretty=format: --name-only --diff-filter=A | sort - | sed '/^$/d'


#=============================================================================#
#                                                                             #
#  88888888ba               88                                                #
#  88      "8b              88                                                #
#  88      ,8P              88                                                #
#  88aaaaaa8P'   ,adPPYba,  88,dPPYba,   ,adPPYYba,  ,adPPYba,   ,adPPYba,    #
#  88""""88'    a8P_____88  88P'    "8a  ""     `Y8  I8[    ""  a8P_____88    #
#  88    `8b    8PP"""""""  88       d8  ,adPPPPP88   `"Y8ba,   8PP"""""""    #
#  88     `8b   "8b,   ,aa  88b,   ,a8"  88,    ,88  aa    ]8I  "8b,   ,aa    #
#  88      `8b   `"Ybbd8"'  8Y"Ybbd8"'   `"8bbdP"Y8  `"YbbdP"'   `"Ybbd8"'    #
#                                                                             #
#=============================================================================#


# ====== #
# SQUASH #
# ====== #

# You can pseudo "squash" all your commits by resetting the index to master,
# and adding all the diffed files in a single commit.
#   Source: "Git: How to squash all commits on branch"
#           https://stackoverflow.com/a/25357146
git checkout yourBranch
git reset $(git merge-base master $(git branch --show-current))
git add -A
git commit -m "one commit on yourBranch"


#=============================================================================#
#                  _   _                                           _          #
#           __ _  (_) | |_            ___   _ __   _   _   _ __   | |_        #
#          / _` | | | | __|  _____   / __| | '__| | | | | | '_ \  | __|       #
#         | (_| | | | | |_  |_____| | (__  | |    | |_| | | |_) | | |_        #
#          \__, | |_|  \__|          \___| |_|     \__, | | .__/   \__|       #
#          |___/                                   |___/  |_|                 #
#                                                                             #
#=============================================================================#

# Install
sudo apt install git-crypt

# Setting up git-crypt repo
# =========================
# make git repo or navigate to your repo
mkdir sample_repo && cd sample_repo
git init

# specify files to encrypt via .gitattributes file
# in repo root
touch .gitattributes
echo "secretfile filter=git-crypt diff=git-crypt" >> .gitattributes
echo "*.key filter=git-crypt diff=git-crypt" >> .gitattributes
git add .gitattributes
git commit -m 'git-crypt attrs'

# dummy files
touch publicfile
touch secretfile
touch nuclear_launch.key


# initialize git-crypt
git-crypt init  # Generating key...

# Now 'secretfile' and any file with ext '.key' are encrypted by git-crypt
git-crypt status
#     encrypted: nuclear_launch.key
# not encrypted: publicfile
#     encrypted: secretfile
# not encrypted: .gitattributes

# Furthermore, the repo is in a "locked" state

#-----------------------------------------------------------------------------#
#                                  NOTE(2025)                                 #
#
#
#
# before you rely on git-crypt again, do make sure that you refamiliarize
# yourself with the git-crypt workflow--specifically the sequence of
# commands.
#
# one thing that is crucially missing from my notes here (which were otherwise
# pretty dec) is A COMMIT.
#
# E.g.:
#   - do you git-crypt lock *before* commit?
#   - is it automatically crypted (matched from .gitattribs)?
#   - because git commit on git-crypt target files will still show the files
#     in clear text locally.
#     - so, does this mean if it was remote, it would be crypted?
#     - or do I need to manually make sure I `git-crypt lock`
#       (after I have already committed the files)
#
#
#   - surely, git-crypt doesn't rely on explicit cmds on target files from the
#     user.
#   - even if so, there should be a simple git-hook integration to make this
#     whole op simple.
#
#
#-----------------------------------------------------------------------------#


# WARNING:
#  you MUST commit changes to .gitattributes before target files
#  are encrypted
git add secretfile
git add nuclear_launch.key


# Configuring users
# =================
# I prefer the shared key approach
git-crypt export-key ../sample_repo_gckey
git-crypt unlock ../sample_repo_gckey

# ADD FILES IN UNLOCKED STATE, then lock

# Encrypting directories
# ======================
# COMMON GOTCHA:
#  simply adding `secret_dir/* filter=git-crypt diff=git-crypt`
#   WILL NOT ENCRYPT ALL CONTENTS IN secret_dir  -- namely, subdirs will not be encrypt
#  to encrypt ALL contents, you must add the following
#  to secret_dir/.gitattributes:
* filter=git-crypt diff=git-crypt
.gitattributes !filter !diff  # this line necessary to insure .gitattributes file is not encrypted


#=============================================================================#
#                    __  __   _                                               #
#                   |  \/  | (_)  _ __   _ __    ___    _ __                  #
#                   | |\/| | | | | '__| | '__|  / _ \  | '__|                 #
#                   | |  | | | | | |    | |    | (_) | | |                    #
#                   |_|  |_| |_| |_|    |_|     \___/  |_|                    #
#                                                                             #
#=============================================================================#
# How I make a "private fork"
#   Reference: https://stackoverflow.com/a/30352360/6880404

# 1. Create a new repo (let's call it `private-repo`) via the Github UI:
# ------------
git clone --bare https://github.com/exampleuser/public-repo.git
cd public-repo.git
git push --mirror https://github.com/yourname/private-repo.git
cd ..
rm -rf public-repo.git


# 2. Clone the private repo so you can work on it:
# ------------
git clone https://github.com/yourname/private-repo.git
cd private-repo
make some changes
git commit
git push origin master

# 3. To pull new hotness from the public repo:
# ------------
cd private-repo
git remote add public https://github.com/exampleuser/public-repo.git
git pull public master # Creates a merge commit
git push origin master
# Awesome, your private repo now has the latest code from the
#  public repo plus your changes!

# 4. Finally, to create a pull request private repo -> public repo:
# -------------
# The only way to create a pull request is to have push access to the
#  public repo.
# This is because you need to push to a branch there
git clone https://github.com/exampleuser/public-repo.git
cd public-repo
git remote add private_repo_yourname https://github.com/yourname/private-repo.git
git checkout -b pull_request_yourname
git pull private_repo_yourname master
git push origin pull_request_yourname
# Now simply create a pull request via the Github UI for public-repo


#=============================================================================#
#          ____    _____   _       _____   _____   ___    ___    _   _        #
#         |  _ \  | ____| | |     | ____| |_   _| |_ _|  / _ \  | \ | |       #
#         | | | | |  _|   | |     |  _|     | |    | |  | | | | |  \| |       #
#         | |_| | | |___  | |___  | |___    | |    | |  | |_| | | |\  |       #
#         |____/  |_____| |_____| |_____|   |_|   |___|  \___/  |_| \_|       #
#                                                                             #
#=============================================================================#

# reference: https://stackoverflow.com/q/2100907/6880404
#------------------------------------------------------------------------------
# Use this to check for large files or dirs
#------------------------------------------------------------------------------
git rev-list --objects --all \
| git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
| sed -n 's/^blob //p' \
| sort --numeric-sort --key=2 \
| cut -c 1-12,41- \
| numfmt --field=2 --to=iec-i --suffix=B --padding=7 --round=nearest

# WARNING: I have only done this as the sole user of a private repo, YMMV

#------------------------------------------------------------------------------
# Removing files from git history
#------------------------------------------------------------------------------
# Removing a file, or directory contents from history completely

#===== Removing a file:
#git filter-branch --force --index-filter 'git rm -rf --ignore-unmatch my_file' --prune-empty --tag-name-filter cat -- --all

#===== Removing all files in my_dir:
#git filter-branch --force --index-filter 'git rm -rf --ignore-unmatch my_dir/*' --prune-empty --tag-name-filter cat -- --all

#===== (What I have actually used):
#git filter-branch --tree-filter 'rm -rf my_dir/*' --prune-empty  -- --all
#git filter-branch --tree-filter 'rm -f stuff.rst more_stuff.md resources.yml *.pdf' --prune-empty  -- --all

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NOTE: THE ABOVE COMMANDS WILL WORK HOWEVER...
#  I just use `git-obliterate` now from the `git-extras` package
#  much simpler. Still follow the commands below

# AFTER EACH filter-branch
rm -rf .git/refs/original

# FINALLY
git reflog expire --expire=now --all
git gc --prune=now
git gc --aggressive --prune=now
#git push --force origin master
git push --all --prune --force  # will delete all branches not in local
# MUST CLONE UPDATED REPO

#------------------------------------------------------------------------------
# Delete misc:
#------------------------------------------------------------------------------
#===== Delete a tag on github
git fetch # if you do not see the tag on local
git tag -d <tag-name>
git push origin :<tag-name>

# if tag name is same as branch:
git tag -d <tag-name>
git push origin :refs/tags/<tag-name>

#==== Delete all local untracked files in repo
git clean -n # to see what will be deleted
git clean -f # to delete

#==== Delete branch
git push --delete <remote_name> <branch_name> # remote
git branch -d <branch_name> # local


#=============================================================================#
#                           _         ____     _____                          #
#                          | |       / __ \   / ____|                         #
#                          | |      | |  | | | |  __                          #
#                          | |      | |  | | | | |_ |                         #
#                          | |____  | |__| | | |__| |                         #
#                          |______|  \____/   \_____|                         #
#                                                                             #
#=============================================================================#

# To disable paging (for log, or ANY git command that uses a pager)
# simply append `--no-pager` immediately after `git` and before the git command.
git --no-pager log

#=== See history for specific file
#    (that may have been deleted)
# If you know the file-path:
git log --all --full-history -- <path-to-file>

# If you don't know:
git log --all --full-history -- "**/thefile.*"

## ^ that should show the list of commits in all branches which touched that
## file.
## You can then find the version of the file you want, and display it:
git show <SHA> -- <path-to-file>

## OR restore it to your working copy:
git checkout <SHA>^ -- <path-to-file>

#---
# There's also `git log -p myfile.py`,
# which shows changes to `myfile.py` over time:
git log -p myfile.py





#=============================================================================#
#                             Changing Authorship                             #
#=============================================================================#


git rev-list $RANGE | while read commit; do
    GIT_COMMITTER_DATE="$(git log -1 --format=%cd --date=iso-strict $commit)" \
    GIT_AUTHOR_DATE="$(git log -1 --format=%ad --date=iso-strict $commit)" \
        git commit --amend -C "$commit" -S --no-edit 2>/dev/null || {
            echo >&2 "Failed to re‑sign commit: $commit"
            exit 1
        }
done


# For all commits in history
# ==========================

git filter-branch --env-filter '
WRONG_EMAIL="foo.bar@A.com"
NEW_NAME="Alice Bob"
NEW_EMAIL="proper.email@foo.com"

if [ "$GIT_COMMITTER_EMAIL" = "$WRONG_EMAIL" ]
then
    export GIT_COMMITTER_NAME="$NEW_NAME"
    export GIT_COMMITTER_EMAIL="$NEW_EMAIL"
fi
if [ "$GIT_AUTHOR_EMAIL" = "$WRONG_EMAIL" ]
then
    export GIT_AUTHOR_NAME="$NEW_NAME"
    export GIT_AUTHOR_EMAIL="$NEW_EMAIL"
fi
' --tag-name-filter cat -- --branches --tags


#=============================================================================#
#                            __  __   _                                       #
#                           |  \/  | (_)  ___    ___                          #
#                           | |\/| | | | / __|  / __|                         #
#                           | |  | | | | \__ \ | (__                          #
#                           |_|  |_| |_| |___/  \___|                         #
#                                                                             #
#=============================================================================#

## If you just want to change the author of your last commit, you can do this:

# Reset your email to the config globally:
git config --global user.email example@email.com

# Now reset the author of your commit without edit required:
git commit --amend --reset-author --no-edit

#To fix my last six commits:
# First set the correct author for current Git repo using git config --local user.name FirstName LastName
# and git config --local user.email first.last@example.com.
# Then apply to the last six commits using:
git rebase --onto HEAD~6 --exec "git commit --amend --reset-author --no-edit" HEAD~6


# Set global git ignore
# ---------------------
git config --global core.excludesfile ~/.Dots/global_gitignore

# Set local git config to ignore untracked files
# ----------------------------------------------
git config --local status.showUntrackedFiles no

# Squash/drop all commits to one commit
# -------------------------------------
git rebase --root -i  # squash or drop for all commits save first


# Get list of largest objects in git repo
#-----------------------------------------
# great for pruning git tree when it becomes bloated with all those
#  notebooks with cell outputs of high-dpi pyplot 3D mplots you
#  forgot to clear
#
# > Credit:
#   User: https://stackoverflow.com/users/380229/raphinesse
#   Link: https://stackoverflow.com/a/42544963/6880404
git rev-list --objects --all \
| git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
| sed -n 's/^blob //p' \
| sort --numeric-sort --key=2 \
| cut -c 1-12,41- \
| numfmt --field=2 --to=iec-i --suffix=B --padding=7 --round=nearest

# This will return something like:
# BLOB-HASH  FILE_SIZE  FILE_NAME
# ...
#6a02a25d7f47   64MiB bigbrainmodel.pth
#99bf17532bef   72MiB iamsmartdev/my_bigass_notebooks/huge_notebook_with_8kres_visualizations.ipynb


# (NOT ALWAYS WORK) To find the branch that committed the blob,
# e.g. `6a02b25d7f47 64MiB bigbrainmodel.pth`
# First, find all commits for the file:
git log --branches -- bigbrainmodel.pth
# commit 56d40e9a9560c125f1ef2aba1dc2c8a68fa7977f
# Author: Hugh Mongus (BigBrain Real ML Engs, Inc.) <ai_influencer420@hotmail.com>
# Date:   Wed April 20 04:20:00 1969 +0300
#
#     Ayyy lmao!



# Script/Cmd to replace all author
# --------------------------------
#!/bin/sh

git filter-branch --env-filter '
OLD_EMAIL="your-old-email@example.com"
CORRECT_NAME="Your Correct Name"
CORRECT_EMAIL="your-correct-email@example.com"
if [ "$GIT_COMMITTER_EMAIL" = "$OLD_EMAIL" ]
then
    export GIT_COMMITTER_NAME="$CORRECT_NAME"
    export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
fi
if [ "$GIT_AUTHOR_EMAIL" = "$OLD_EMAIL" ]
then
    export GIT_AUTHOR_NAME="$CORRECT_NAME"
    export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
fi
' --tag-name-filter cat -- --branches --tags


# Add a new remote
# ================
# First, verify current remotes:
git remote -v

# Add your new remote:
git remote add my_new_remote git@github.com:evdcush/FooAndBar.git


#=============================================================================#
#                                                                             #
#                      ████████  █████   ██████  ███████                      #
#                         ██    ██   ██ ██       ██                           #
#                         ██    ███████ ██   ███ ███████                      #
#                         ██    ██   ██ ██    ██      ██                      #
#                         ██    ██   ██  ██████  ███████                      #
#                                                                             #
#=============================================================================#

# Create a local tag:
git tag <tag-name>

# Delete it:
git tag -d <tag-name>

# Get all tags from remote:
git fetch --tags



#=============================================================================#
#                                                                             #
#            ██████  ██ ████████       ████████ ██ ██████  ███████            #
#           ██       ██    ██             ██    ██ ██   ██ ██                 #
#           ██   ███ ██    ██    █████    ██    ██ ██████  ███████            #
#           ██    ██ ██    ██             ██    ██ ██           ██            #
#            ██████  ██    ██             ██    ██ ██      ███████            #
#                                                                             #
# https://github.com/git-tips/tips                                            #
#=============================================================================#

# Remove sensitive data from history, after a push
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch <path-to-your-file>' \
--prune-empty --tag-name-filter cat -- --all && \
git push origin --force --all

# Sync with remote, overwrite local changes.
git fetch origin && git reset --hard origin/master && git clean -f -d

# List of all files changed in a commit.
git diff-tree --no-commit-id --name-only -r <commit-ish>

# Delete remote branch.
git push origin --delete <remote_branchname>



#=============================================================================#
#                                                                             #
#          d8b 888                          888                               #
#          Y8P 888                          888                               #
#              888                          888                               #
#  .d88b.  888 888888      .d88b.  888  888 888888 888d888  8888b.  .d8888b   #
# d88P"88b 888 888        d8P  Y8b `Y8bd8P' 888    888P"       "88b 88K       #
# 888  888 888 888        88888888   X88K   888    888     .d888888 "Y8888b.  #
# Y88b 888 888 Y88b.      Y8b.     .d8""8b. Y88b.  888     888  888      X88  #
#  "Y88888 888  "Y888      "Y8888  888  888  "Y888 888     "Y888888  88888P'  #
#      888                                                                    #
# Y8b d88P                                                                    #
#  "Y88P"                                                                     #
#                                                                             #
#=============================================================================#
# sudo apt install git-extras
#
# deetz, cur ver
# ==============
# Package: git-extras
# Version: 6.1.0-1
# Section: universe/vcs
# Depends: git
# Homepage: https://github.com/tj/git-extras
# Description-en: Extra commands for git
#  This package provides extra git commands to easily solve typical tasks such as
#  managing remote branches, managing feature/refactor/bug workflows and generate
#  some repository stats.


NAME
       git-extras - Awesome GIT utilities

SYNOPSIS
       git-extras [-v,--version] [-h,--help] [update]

OPTIONS
       -v, --version

       Show git-extras version number.

       -h, --help

       Show this help. This option can also be used for any of the extras commands.

       update

       Self update.

ENVIRONMENT AND CONFIGURATION VARIABLES
       git config --add git-extras.default-branch $BRANCH

       Change the default branch to $BRANCH (defaut to master).

COMMANDS
       ○   git-alias(1) Define, search and show aliases

       ○   git-archive-file(1) Export the current HEAD of the git repository to an archive

       ○   git-authors(1) Generate authors report

       ○   git-browse(1) View the web page for the current repository

       ○   git-brv(1) List branches sorted by their last commit date

       ○   git-bulk(1) Run git commands on multiple repositories

       ○   git-changelog(1) Generate a changelog report

       ○   git-clear-soft(1) Soft clean up a repository

       ○   git-clear(1) Rigorously clean up a repository

       ○   git-coauthor(1) Add a co-author to the last commit

       ○   git-commits-since(1) Show commit logs since some date

       ○   git-contrib(1) Show user´s contributions

       ○   git-count(1) Show commit count

       ○   git-cp(1) Copy a file keeping its history

       ○   git-create-branch(1) Create branches

       ○   git-delete-branch(1) Delete branches

       ○   git-delete-merged-branches(1) Delete merged branches

       ○   git-delete-submodule(1) Delete submodules

       ○   git-delete-tag(1) Delete tags

       ○   git-delta(1) Lists changed files

       ○   git-effort(1) Show effort statistics on file(s)

       ○   git-feature(1) Create/Merge feature branch

       ○   git-force-clone(1) overwrite local repositories with clone

       ○   git-fork(1) Fork a repo on github

       ○   git-fresh-branch(1) Create fresh branches

       ○   git-gh-pages(1) Create the GitHub Pages branch

       ○   git-graft(1) Merge and destroy a given branch

       ○   git-guilt(1) calculate change between two revisions

       ○   git-ignore-io(1) Get sample gitignore file

       ○   git-ignore(1) Add .gitignore patterns

       ○   git-info(1) Returns information on current repository

       ○   git-local-commits(1) List local commits

       ○   git-lock(1) Lock a file excluded from version control

       ○   git-locked(1) ls files that have been locked

       ○   git-merge-into(1) Merge one branch into another

       ○   git-merge-repo(1) Merge two repo histories

       ○   git-missing(1) Show commits missing from another branch

       ○   git-mr(1) Checks out a merge request locally

       ○   git-obliterate(1) rewrite past commits to remove some files

       ○   git-paste(1) Send patches to pastebin for chat conversations

       ○   git-pr(1) Checks out a pull request locally

       ○   git-psykorebase(1) Rebase a branch with a merge commit

       ○   git-pull-request(1) Create pull request for GitHub project

       ○   git-reauthor(1) Rewrite history to change author´s identity

       ○   git-rebase-patch(1) Rebases a patch

       ○   git-release(1) Commit, tag and push changes to the repository

       ○   git-rename-branch(1) rename local branch and push to remote

       ○   git-rename-remote(1) Rename a remote

       ○   git-rename-tag(1) Rename a tag

       ○   git-repl(1) git read-eval-print-loop

       ○   git-reset-file(1) Reset one file

       ○   git-root(1) show path of root

       ○   git-scp(1) Copy files to SSH compatible git-remote

       ○   git-sed(1) replace patterns in git-controlled files

       ○   git-setup(1) Set up a git repository

       ○   git-show-merged-branches(1) Show merged branches

       ○   git-show-tree(1) show branch tree of commit history

       ○   git-show-unmerged-branches(1) Show unmerged branches

       ○   git-squash(1) Import changes from a branch

       ○   git-stamp(1) Stamp the last commit message

       ○   git-standup(1) Recall the commit history

       ○   git-summary(1) Show repository summary

       ○   git-sync(1) Sync local branch with remote branch

       ○   git-touch(1) Touch and add file to the index

       ○   git-undo(1) Remove latest commits

       ○   git-unlock(1) Unlock a file excluded from version control

AUTHOR
       Written by Tj Holowaychuk <tj@vision-media.ca>

REPORTING BUGS
       <https://github.com/tj/git-extras/issues>

SEE ALSO
       <https://github.com/tj/git-extras>

                                                                            October 2020                                                               GIT-EXTRAS(1)
