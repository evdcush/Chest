#!/usr/bin/env python
import os
import subprocess

# Get paths
# =========
HOME = os.environ['HOME']
src_path  = os.path.abspath(os.path.dirname(__file__))
dest_path = f'{HOME}/.'

# dotfiles
# ========
ln_to_dest_files = ['aliases', 'functions', 'paths']
ln_to_home_files = ['zshrc']


def symlink_files_src2dest(*files, src=src_path, dest=dest_path):
    if not os.path.exists(dest): os.makedirs(dest)
    for src_fname in files:
        #==== format paths and symlink
        src_file_path  = f'{src}/{src_fname}'
        dest_file_path = f'{dest}/{src_fname}'
        subprocess.run(f'ln -sf {src_file_path} {dest_file_path}', shell=True)

        #==== sanity check
        if not os.path.exists(dest_file_path):
            print(f'\nERROR: {src_fname} did not successfully symlink '
                  f'to {dest_file_path}')


if __name__ == '__main__':
    symlink_files_src2dest(*ln_to_dest_files, dest=dest_path + 'Dots')
    symlink_files_src2dest(*ln_to_home_files)
    print('\nAll files processed')
