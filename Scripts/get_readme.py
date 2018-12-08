#!/usr/bin/env python

import os
import sys
import subprocess as sbp

# dest dirs
HOME = os.environ['HOME']
dest    = f'{HOME}/.Nextcloud/READMEs'
ln_dest = f'{HOME}/Documents/READMEs'

def wget_readme(url, dest=dest):
    # README wget formatting
    readme_fname = f'{url.split("/")[-1]}.md'
    u = url.replace('github', 'raw.githubusercontent', 1) + '/master/README.md'

    # wget and symlink
    sbp.run(f'wget {u} -O {dest}/{readme_fname}', shell=True)
    sbp.run(f'ln -sf {dest}/{readme_fname} {ln_dest}', shell=True)

if __name__ == '__main__':
    URL = sys.argv[1]
    wget_readme(URL)


