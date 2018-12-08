#!/usr/bin/env python

import os
import sys
import subprocess as sbp

# dest dirs
dest = f"{os.environ['HOME']}/.Nextcloud/READMEs"

def wget_readme(url, dest=dest):
    # README wget formatting
    readme_fname = f'{url.split("/")[-1]}.md'
    u = url.replace('github', 'raw.githubusercontent', 1) + '/master/README.md'
    fpath = f'{dest}/{readme_fname}'
    if os.path.exists(fpath):
        print(f'DUPLICATE; existent {readme_fname}')
        sbp.run(f"mv {fpath} {fpath[:-3] + '.prev.md'}", shell=True)

    # wget and symlink
    sbp.run(f'wget {u} -O {fpath}', shell=True)

if __name__ == '__main__':
    URL = sys.argv[1]
    if len(sys.argv) > 2:
        dest = sys.argv[2]
    wget_readme(URL, dest)


