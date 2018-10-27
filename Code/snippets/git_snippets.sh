#!/bin/bash

#  __  __   _____   _____    _____     ____    _____
# |  \/  | |_   _| |  __ \  |  __ \   / __ \  |  __ \
# | \  / |   | |   | |__) | | |__) | | |  | | | |__) |
# | |\/| |   | |   |  _  /  |  _  /  | |  | | |  _  /
# | |  | |  _| |_  | | \ \  | | \ \  | |__| | | | \ \
# |_|  |_| |_____| |_|  \_\ |_|  \_\  \____/  |_|  \_\
#
###############################################################################
#  _ _  ___  ___  ___ __   __   _    _____  ___    ___   ___   ___  _  __ _ _
# ( | )| _ \| _ \|_ _|\ \ / /  /_\  |_   _|| __|  | __| / _ \ | _ \| |/ /( | )
#  V V |  _/|   / | |  \ V /  / _ \   | |  | _|   | _| | (_) ||   /| ' <  V V
#      |_|  |_|_\|___|  \_/  /_/ \_\  |_|  |___|  |_|   \___/ |_|_\|_|\_\
#
#------------------------------------------------------------------------------
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




#==============================================================================



###############################################################################
#  _____    ______   _        ______   _______   _____   _   _    _____
# |  __ \  |  ____| | |      |  ____| |__   __| |_   _| | \ | |  / ____|
# | |  | | | |__    | |      | |__       | |      | |   |  \| | | |  __
# | |  | | |  __|   | |      |  __|      | |      | |   | . ` | | | |_ |
# | |__| | | |____  | |____  | |____     | |     _| |_  | |\  | | |__| |
# |_____/  |______| |______| |______|    |_|    |_____| |_| \_|  \_____|
#
###############################################################################
# reference: https://stackoverflow.com/q/2100907/6880404

# WARNING: I have only done this as the sole user of a private repo, YMMV

# Removing files from git history
#--------------------------------
# Removing a file, or directory contents from history completely

#===== Removing a file:
git filter-branch --force --index-filter 'git rm -rf --ignore-unmatch my_file' --prune-empty --tag-name-filter cat -- --all

#===== Removing all files in my_dir:
git filter-branch --force --index-filter 'git rm -rf --ignore-unmatch my_dir/*' --prune-empty --tag-name-filter cat -- --all

#===== (What I have actually used:)
git filter-branch --tree-filter 'rm -rf my_dir/*' --prune-empty  -- --all
git filter-branch --tree-filter 'rm -f topics.rst glossary.rst itinerary.md resources.rst *.pdf README.md' --prune-empty  -- --all

# AFTER EACH filter-branch
rm -rf .git/refs/original
git push --force origin master

# FINALLY
git reflog expire --expire=now --all
git gc --prune=now
git gc --aggressive --prune=now



###############################################################################
#  _____    ______   __  __    ____    _______   ______
# |  __ \  |  ____| |  \/  |  / __ \  |__   __| |  ____|
# | |__) | | |__    | \  / | | |  | |    | |    | |__
# |  _  /  |  __|   | |\/| | | |  | |    | |    |  __|
# | | \ \  | |____  | |  | | | |__| |    | |    | |____
# |_|  \_\ |______| |_|  |_|  \____/     |_|    |______|
#
###############################################################################


#------------------------------------------------------------------------------
#  ___   ___     _     _  _    ___   _  _
# | _ ) | _ \   /_\   | \| |  / __| | || |
# | _ \ |   /  / _ \  | .` | | (__  | __ |
# |___/ |_|_\ /_/ \_\ |_|\_|  \___| |_||_|
#
#------------------------------------------------------------------------------

# push a local branch
git push <remote-name> <local-branch-name>:<remote-branch-name>

# To pull all new remote branches from remote:
git remote update


https://stackoverflow.com/questions/2765421/how-do-i-push-a-new-local-branch-to-a-remote-git-repository-and-track-it-too

https://stackoverflow.com/questions/1783405/how-do-i-check-out-a-remote-git-branch

https://stackoverflow.com/questions/1519006/how-do-you-create-a-remote-git-branch

https://stackoverflow.com/questions/24301914/git-create-local-branch-from-existing-remote-branch



###############################################################################
#    __  __               _____    _____    _____
#   |  \/  |     /\      / ____|  |_   _|  / ____|
#   | \  / |    /  \    | |  __     | |   | |
#   | |\/| |   / /\ \   | | |_ |    | |   | |
#   | |  | |  / ____ \  | |__| |   _| |_  | |____
#   |_|  |_| /_/    \_\  \_____|  |_____|  \_____|
#
###############################################################################


# Get list of largest objects in git repo
#-----------------------------------------
# > Credit:
#   User: https://stackoverflow.com/users/380229/raphinesse
#   Link: https://stackoverflow.com/a/42544963/6880404
git rev-list --objects --all \
| git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
| sed -n 's/^blob //p' \
| sort --numeric-sort --key=2 \
| cut -c 1-12,41- \
| numfmt --field=2 --to=iec-i --suffix=B --padding=7 --round=nearest
