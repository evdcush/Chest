import os
import subprocess


# Nifty helpers
# =============
file_exists = lambda file_path: os.path.exists(file_path)

def make_dir(my_dir):
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)

def subrun(cmd, **kwargs):
    print(f'running "{cmd}"')
    subprocess.run(cmd, shell=True, **kwargs)


# Get paths
# =========
#---- base paths
src_path = os.path.abspath(os.path.dirname(__file__))
HOME = os.environ['HOME']
shell_env_rc = f'{HOME}/.zshrc'

#---- link destination
dest_path = f'{HOME}/.Dots'

#---- dotfiles
ln_to_dest_files = ['aliases', 'functions', 'paths']
ln_to_home_files = ['zshrc']


# Link files
# ==========
#---- symlink format
symlink_cmd = "ln -sf {src} {dest}"
#---- symlink func
def symlink_files_src2dest(*files, src=src_path, dest=dest_path):
    make_dir(dest)
    # only $home dotfiles get actual dot
    prefix = '.' if dest == HOME else ''

    for src_fname in files:
        #==== fully formatted symlink paths
        dest_fname = prefix + src_fname
        src_file_path  = f'{src}/{src_fname}'
        dest_file_path = f'{dest}/{dest_fname}'

        #==== symlink time!
        cmd = symlink_cmd.format(src=src_file_path, dest=dest_file_path)
        subrun(cmd)

        #==== sanity check
        if not file_exists(dest_file_path):
            print(f'\nERROR: {src_fname} did not successfully symlink '
                  f'to {dest_file_path}')



if __name__ == '__main__':
    symlink_files_src2dest(*ln_to_dest_files)
    symlink_files_src2dest(*ln_to_home_files, dest=HOME)
    print('All files processed')
