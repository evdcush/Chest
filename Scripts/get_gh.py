#!/usr/bin/env python

import os
import sys
import yaml
import subprocess as sbp

# Load conf
HOME = os.environ['HOME']

with open('conf.yml') as yml:
    conf = yaml.load(yml)
nsync_path = conf['nsync']['path'].replace('~', HOME)

# dest dirs
readme_dest = f"{nsync_path}{conf['nsync']['readme_dir']}"
file_dest = f"{HOME}/Projects"

def check_dup(file_path):
    if os.path.exists(file_path):
        print(f'Duplicate existent for {file_path}\nMoving previous to .cache')
        sbp.run(f'mv {file_path} {HOME}/.cache', shell=True)


def wget_raw(url, fname, dest=file_dest):
    # Format endpoints
    u = url.replace('github', 'raw.githubusercontent', 1) + f'/master/{fname}'
    fpath = f'{dest}/{fname}'
    check_dup(fpath)

    # wget and symlink
    sbp.run(f'wget {u} -O {fpath}', shell=True)

# Potential readme file names:
rdme = ['README.md', 'README', 'README.rst', 'README.txt',
        'readme.md', 'readme', 'readme.txt']

def wget_readme(url, dest=readme_dest):
    chk_exist = lambda u: sbp.run(f'wget --spider {u}', shell=True)

    # URL formatting
    u = url.replace('github', 'raw.githubusercontent', 1) + '/master/'
    fname = f'{url.split("/")[-1]}'

    # Check for filename format
    correct_rdme_name = ''
    for rdm in rdme:
        if chk_exist(u + rdm).returncode == 0:
            correct_rdme_name = rdm
            break
    if correct_rdme_name == '':
        print('readme does not match standard readme file names or does not exist')
    else:
        # Get extension of correct readme
        ext = correct_rdme_name.split('.')[-1]
        if ext != correct_rdme_name:
            fname += '.' + ext
        #=== append true readme name to url
        u = u + correct_rdme_name
        #=== append file write name to dest
        dest = f'{dest}/{fname}'
        check_dup(dest)
        #=== Get readme
        sbp.run(f'wget {u} -O {dest}', shell=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    URL = args.pop()
    if args:
        # args is [filename, dest] or [filename]
        wget_raw(URL, *args)
    else:
        wget_readme(URL)

