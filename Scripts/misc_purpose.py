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
#-----------------------------------------------------------------------------#
#                               PyPi API stuff                                #
#-----------------------------------------------------------------------------#
"""
NOTE: the PyPi API no longer provides download numbers
"""

import requests

api_res = requests.get('https://pypi.org/pypi/numpy/json')
res_json = api_res.json()
res_json.keys() # dict_keys(['info', 'last_serial', 'releases', 'urls'])

print(res_json['info'])
{'author': 'Travis E. Oliphant et al.',
 'author_email': '',
 'bugtrack_url': None,
 'classifiers': ['Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved',
                 'Operating System :: MacOS',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Programming Language :: C',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: Implementation :: CPython',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Software Development'],
 'description': 'It provides:\n'
                '\n'
                '- a powerful N-dimensional array object\n'
                '- sophisticated (broadcasting) functions\n'
                '- tools for integrating C/C++ and Fortran code\n'
                '- useful linear algebra, Fourier transform, and random number '
                'capabilities\n'
                '- and much more\n'
                '\n'
                'Besides its obvious scientific uses, NumPy can also be used '
                'as an efficient\n'
                'multi-dimensional container of generic data. Arbitrary '
                'data-types can be\n'
                'defined. This allows NumPy to seamlessly and speedily '
                'integrate with a wide\n'
                'variety of databases.\n'
                '\n'
                'All NumPy wheels distributed on PyPI are BSD licensed.\n'
                '\n'
                '\n'
                '\n',
 'description_content_type': '',
 'docs_url': None,
 'download_url': 'https://pypi.python.org/pypi/numpy',
 'downloads': {'last_day': -1, 'last_month': -1, 'last_week': -1},
 'home_page': 'https://www.numpy.org',
 'keywords': '',
 'license': 'BSD',
 'maintainer': 'NumPy Developers',
 'maintainer_email': 'numpy-discussion@python.org',
 'name': 'numpy',
 'package_url': 'https://pypi.org/project/numpy/',
 'platform': 'Windows',
 'project_url': 'https://pypi.org/project/numpy/',
 'project_urls': {'Download': 'https://pypi.python.org/pypi/numpy',
                  'Homepage': 'https://www.numpy.org'},
 'release_url': 'https://pypi.org/project/numpy/1.16.2/',
 'requires_dist': None,
 'requires_python': '>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
 'summary': 'NumPy is the fundamental package for array computing with Python.',
 'version': '1.16.2'}
