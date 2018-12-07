import os
import sys
import subprocess

HOME = os.environ['HOME']

# wget target formatting helpers
gh_rep = lambda u: u.replace('github.com', 'raw.githubusercontent.com')
add_suffix = lambda u: u + '/master/README.md'

# dest dir
dest    = f'{HOME}/.Nextcloud/READMEs'
ln_dest = f'{HOME}/Documents/READMEs'
get_file_name = lambda u: u.split('/')[-1]

# subprocess wrapper
def subrun(cmd, **kwargs):
    print(f'\n\nsubrun: {cmd}\n\n')
    subprocess.run(cmd, shell=True, **kwargs)

def wget_readme(url, dest=dest):
    # README wget
    readme_fname = get_file_name(url) + '.md'
    arg = add_suffix(gh_rep(url))
    wget_cmd = f'wget {arg} -O {dest}/{readme_fname}'

    #==== symlink cmd
    ln_cmd = f'ln -sf {dest} {ln_dest}'

    # subprocess calls
    subrun(wget_cmd)
    subrun(ln_cmd)

if __name__ == '__main__':
    URL = sys.argv[1]
    wget_readme(URL)


