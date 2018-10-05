fromt pprint import PrettyPrinter

pretty_printer = PrettyPrinter()
pp = pretty_printer.pprint


class MyPipPackages:
    packages = []
    def print_pkgs(self, key=None):
        for package in self.packages:
            for pkg_name, pkg in package.items():
                pp(pkg)

    def process_strings(self, pkg_dict):
        for k, v in pkg_dict.items():
            v = v.strip().split('\n')

    def __call__(self, pkg_dict):
        processed_dict = self.process_strings(pkg_dict)
        self.packages.append(processed_dict)



#pip install -U pip setuptools wheel

#============ Science essentials

FOUNDATION = {'FOUNDATION':"""
numpy
scipy
sklearn
matplotlib


""",
'order': 0,}

#------------- Science misc

sci_misc = {'sci_misc':"""
autograd
sympy
scikit-video
pydataset

""",
'order': 100,
}




#============ AI libraries/frameworks
AI = {'AI':"""
chainer
gym
chainerrl
cupy
neupy


""",
'order': 1,
}


# TF dependencies
tf_deps = {'tf_deps':"""
keras
keras-applications
keras-preprocessing
protobuf


""",
'order': 100,
}


#-------------- Starcraft
starcraft = {'starcraft':"""
pygame
pysc2
sc2
scbw

""",
'order': 50,
'prereq': True,
}

#============ Notebook
notebook = {'notebook':"""
ipython
jupyter
jupyterlab
jupyterthemes


""",
'order': 5,
}



#============ Applications
apps = {'apps':"""
retext


""",
'order': 100,
}


#============ Utiltities
#-------------- Dev
utils = {'utils':"""
apt-select
bs4
black
bidict
binarytree
cookiecutter
h5py
hickle
html5lib
pyqt5
pillow
cookiecutter
nose
pygame
pycurl
pyviz


""",
'order': 40,
}

D = {'D':"""



""",
'order': ,
}
#-------------- Pip/PyPi/Python-package related
pip_related = {'pip_related':"""
pypi-cli
pip-autoremove
vermin
pydeps
pipdeptree


""",
'order': 10,
}



#-------------- Text/media
text_media = {'text_media':"""
sphinx
sphinx-rtd-theme
docutils
doc2dash
pygments
bibtexparser
pandocfilters
cairosvg
betterbib
pyfiglet
numpydoc


""",
'order': 50,
}


#============ CLI/Shell


cli = {'cli':"""
docopt
click
thefuck


""",
'order': 25,
}
#-------------- Parsers
#docopt
#click
#
##-------------- CLI utils
#thefuck





#============ misc
misc = {'misc':"""
art
pyfiglet
starred
oh-my-stars
termdown

""",
'order': 80,
}


#============= Haven't really tried
look_into = {'look_into':"""
buku
gitsuggest
rstwatch
rstdiary
mccabe
bidict
binarytree
numpydoc
pelican
weasyprint
pybtex
grip

""",
'order': 100,
}


#============= experiment with, never installed
look_into = {'look_into':"""
""",
'order': 100,
}
python-box
github-sectory
astrodash # spectral convolutions deep learning supernovae
pipsalabim  # guesses your dependencies, no need for req file
dbcollection # collection of popular ML datasets
