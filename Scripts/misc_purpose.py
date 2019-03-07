"""
Preservation of one-off scripts used for some purpose, that
might be useful again some time.

"""





#██████████████████████████████████████████████████████████████████████████████
#-----------------------------------------------------------------------------#
#                             Combining src files                             #
#-----------------------------------------------------------------------------#

from glob import glob

task_dirs  = {d.split('/')[-1]: [] for d in glob('common-tasks/*')}
task_fpaths = glob('common-tasks/*/*')

for fp in task_fpaths:
    key = fp.split('/')[1]
    task_dirs[key].append(fp)

concatted_fname = 'common_tasks.cpp'


line = '//' + '='*76 + '//'

soft_line = "\n// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n"

def get_header_line(topic):
    header = f"\n\n{line}\n// {topic:^74} //\n{line}\n\n"
    return header


def write_file():
    with open(concatted_fname, 'w') as wfile:
        for k, v in task_dirs.items():
            cur_header = get_header_line(k)
            wfile.write(cur_header)
            for fpath in v:
                with open(fpath) as cpp_file:
                    for line in cpp_file.read():
                        wfile.write(line)
                wfile.write(soft_line)

#██████████████████████████████████████████████████████████████████████████████
#-----------------------------------------------------------------------------#
#                Extracting all pdf copies attached in Zotero                 #
#-----------------------------------------------------------------------------#

import os
import glob
import shutil

# paths
dst  = os.environ['HOME'] + '/LITERATURE'
srcs = [os.path.abspath(g) for g in glob.glob('./**/*.pdf', recursive=True)]

# copy
def srcs2dst(srcs, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for s in srcs:
        sname = s.split('/')[-1]
        shutil.copy(s, dst)
        print(f'Moved {sname}')

#srcs2dst(srcs, dst)

#██████████████████████████████████████████████████████████████████████████████
