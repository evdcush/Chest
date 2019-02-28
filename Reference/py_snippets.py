###############################################################################
#####################   make script executable bin   ##########################
"""
# How to make your 'poop.py' script executable bin

1. put this at the top of script:
    #!/usr/bin/env python

2. `chmod +x poop.py`  # you are already finished at this step

3. ln -sf poop.py dir/thats/in/PATH/poop

# Call like you would other bins from cli:
$ poop
"pooping in the toilet"

$ poop airplane
"pooping in the airplane"

"""


###############################################################################
######################    Numpy, restore shape      ###########################

def restore_axis_shape(x, ax, d):
    """ Restores an axis ax that was reduced from x
    to it's original shape d
    Just a wrapper for the tedious
    broadcast_to(expand_dims(...)...) op

    Assumes
    -------
        x.ndim >= 1
        ax : int  (only one dim being restored through ax)
    """
    assert x.ndim >= 1 and isinstance(ax, int)

    # Restore shape
    bcast_shape = x.shape[:ax] + (d,) + x.shape[ax:]
    return np.broadcast_to(np.expand_dims(x, ax), bcast_shape)


###############################################################################
#######################       Relative imports      ###########################
import os
import sys

# TO GET TRUE FILE DIRECTORY PATH, USE:
os.path.abspath(os.path.dirname(__file__))
#----> EVERYTHING ELSE does pathing from caller path

# GA <---- dataset
#=======================
fpath = os.path.abspath(os.path.dirname(__file__))
path_to_dataset = fpath.rstrip(fpath.split('/')[-1]) + 'data'
sys.path.append(path_to_dataset)


###############################################################################
######################           DOWNLOADING        ###########################
import os
import urllib.request

urls = ["http://www.irs.gov/pub/irs-pdf/f1040.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040a.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040ez.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040es.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040sb.pdf"]

def download_uo(url):
    """
    A coroutine to download the specified url
    """
    request = urllib.request.urlopen(url)
    filename = os.path.basename(url)
    with open(filename, 'wb') as file_handle:
        while True:
            chunk = request.read(1024)
            if not chunk:
                break
            file_handle.write(chunk)
    msg = 'Finished downloading {filename}'.format(filename=filename)
    return msg

def download_ur(url):
    """
    A coroutine to download the specified url
    """
    #request = urllib.request.urlopen(url)
    filename = os.path.basename(url)
    req = urllib.request.urlretrieve(url, filename)
    msg = 'Finished downloading {filename}'.format(filename=filename)
    return msg

import time

def runner(fn):
    tstart = time.time()
    for u in urls:
        fn(u)
    #print(f"time elapsed: {time.time() - tstart}s")

# time diff is trivial for sample urls, but urlretrieve slightly faster



###############################################################################
########################         ARGPARSE          ############################
###############################################################################
import argparse

# To use arg as bool flag AND possible value variable:
parser.add_argument('-s', '--save',
                    nargs='?',
                    default='',
                    const='~')
"""
# NO SAVE (interpretted from default)
$ python aparse.py
Namespace(save='')

# SAVE DEFAULT, but to the specified "default" (const) location
$ python aparse.py -s
Namespace(save='~')

# SAVE CUSTOM
$ python aparse.py -s "$PWD"
Namespace(save='/home/user/path/to/current_dir')
"""

# At a glance
# -----------
# in a script aparse.py:
parser = argparse.ArgumentParser(description='Processing integers.') # description of program
parser.add_argument('integers',  # arg name
                    metavar='N', # how the arg is referred in help msg
                    type=int,    # type conversion
                    nargs='+',   # indicates 1 or more vals to be passed to arg
                    help='integer args') # help msg

if __name__ == '__main__':
    print(parser.parse_args())

"""
>>> python aparse.py -h
usage: aparse.py [-h] N [N ...]

Processing integers.

positional arguments:
  N           integer args

optional arguments:
  -h, --help  show this help message and exit

>>> python aparse.py
usage: aparse.py [-h] N [N ...]
aparse.py: error: the following arguments are required: N

>>> python aparse.py 14
Namespace(integers=[14])

>>> python aparse.py 3 4 5 1 2
Namespace(integers=[3, 4, 5, 1, 2])

"""

# On the 'nargs' argument
# -----------------------
"""
- '?' and '*' can both fall back on a default if not values
- '*' and '+' always store elements in a list; '?' is not a list
"""
nargs='?' # means 1 or None values
# >>> python aparse.py
# Namespace(integers=None)

nargs='+' # means 1 or more values
# >>> python aparse.py
# aparse.py: error: the following arguments are required: N

nargs='*' # means 0 or more values
# >>> python aparse.py
# Namespace(integers=[])


# Actions
# =======
# STORE CONSTANT
parser.add_argument('-f', '--foo',
                    action='store_constant',
                    const='Bar')
# >>> python aparse.py
# Namespace(foo=None)
# >>> python aparse.py -f
# Namespace(foo='Bar')
# >>> python aparse.py -f 'Hello World'
# aparse.py: error: unrecognized arguments: Hello World

# APPEND
parser.add_argument('--baz', action='append')
# >>> python aparse.py
# aparse.py: error: argument -b/--baz: expected one argument
# >>> python aparse.py --baz 'hello'
# Namespace(baz=['hello'])
# >>> python aparse.py --baz 'hello' --baz 'there' --baz 'friend'
# Namespace(baz=['hello', 'there', 'friend'])


#------------------------------------------------------------------------------
#                   ARGPARSE FROM A CONFIG FILE
#           get args and defaults from a yaml or json file
#------------------------------------------------------------------------------

class Parser:
    def __init__(self, config_path=config_path):
        self.argparser = argparse.ArgumentParser()
        self.config_path = config_path
        self.process_config()

    def process_config(self):
        with open(self.config_path) as conf:
            config = yaml.load(conf)
            for key, val in config.items():
                self.argparser.add_argument(f'--{key}', default=val)

    def parse(self):
        self.args = self.argparser.parse_args()
        return self.args




###############################################################################
########################       MY DECORATORS       ############################
from functools import wraps

# BASIC TEMPLATE
def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # call decorated func
        return func(*args, **kwargs)
    return wrapper


# Nifty decorator to both comment and prevent running unfinished funcs
def TODO(f):
    """ Serves as a convenient, clear flag for developers and insures
        wrapee func will not be called """
    @wraps(f)
    def not_finished():
        print('\n  {} IS INCOMPLETE'.format(f.__name__))
    return not_finished


def NOTIMPLEMENTED(f):
    """ Like TODO, but for functions in a class
        raises error when wrappee is called """
    @wraps(f)
    def not_implemented(*args, **kwargs):
        func_class = args[0]
        f_class_name = func_class.get_class_name()
        f_name = f.__name__
        msg = '\n  Class: {}, function: {} has not been implemented!\n'
        print(msg.format(f_class_name, f_name))
        raise NotImplementedError
    return not_implemented


def INSPECT(f):
    """ Will interupt code execution and enter an interactive shell
    where you can test out the input/output values.

    (it's even better just to put that code.interact line directly
    before the line in the actual function you want to inspect)
    """
    @wraps(f)
    def inspector(*args, **kwargs):
        print('\n Inspecting function: <{}>'.format(f.__name__))
        x = args
        y = kwargs
        z = f(*args, **kwargs)
        code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        return z
    return inspector



# Class method decorators
#------------------------
# Yes, that's right, they can access/mutate class instances through self!

def preserve_default(method):
    """ always saves method inputs to fn_vars """
    @wraps(method)
    def preserve_args(self, *args, **kwargs):
        self.fn_vars = args
        return method(self, *args, **kwargs)
    return preserve_args


def preserve(inputs=True, outputs=True):
    """ Preserve inputs and outputs to fn_vars
    Also provides kwargs for individual cases
     - sometimes you only need inputs, or outputs
       this decorator allows you to choose which
       parts of the function you want to preserve
    """
    def inner_preserve(method):
        """ outer preserve is like a decorator to this
        decorator
        """
        @wraps(method)
        def preserve_args(self, *args, **kwargs):
            my_args = ()
            if inputs:
                my_args += args
            ret = method(self, *args, **kwargs)
            if outputs:
                my_args += (ret,)
            self.fn_vars = my_args
            return ret
        return preserve_args
    return inner_preserve


###############################################################################
########################       Special chars       ############################

py_lits = {'Bell a':'\a', 'Backspace b':'\b', 'Formfeed f':'\f',
           'Linefeed n':'\n', 'Carriage Return r':'\r',
           'Horizontal tab t':'\t', 'Vertical tab v':'\v'}

# Difference Between
#  - newline char "\n"
#  - carriage return char "\r"
#=================================================
"""
# Both are control characters that control printing/write of text.
  * They both move the 'cursor' or RW pointer wrt the console,
    or whatever output device is being used
BUT
* '\n' moves the cursor to the *next line* of the console
* '\r' moves the cursor to the *BEGINNING* of the current line in console
"""
NEWLINE = '\n'
print("Hello \nWorld")
'Hello'
'World'

print("Hell\no World")
'Hell'
'o World'


CARRIAGE_RETURN = '\r'
print("Hello \rWorld")
'World'

print("Hell\ro World")
'o World'

#========== Using '\r' to make a simple animation

import time
import sys

animation = '|/-\|'

for i in range(100):
    time.sleep(0.1)
    sys.stdout.write("\r" + animation[i % len(animation)])
    sys.stdout.flush()
print('End!')


###############################################################################
########################     Access parent attr    ############################

class Foo:
    def __init__(self):
        self._cache = None

    @property
    def cache(self):
        if self._cache is not None:
            tmp_obj = self._cache
            self._cache = None
            return tmp_obj
        #return self._cache

    @cache.setter
    def cache(self, x):
        self._cache = x

    def boo(self, x):
        print(f'BOO! haha, got {x} thx')
        self.cache = x

    def poo(self):
        print('going poop, you can have your x back')
        x = self.cache
        return x



class Baz(Foo):
    @property
    def cache(self):
        foo_cache = Foo.cache.fget(self)
        if foo_cache is not None:
            return foo_cache

    @cache.setter
    def cache(self, x):
        Foo.cache.fset(self, x)
        self.xlen = len(x)

    def bar(self):
        xlen = self.xlen
        self.xlen = None
        print(f'in bar, here is your silly xlen: {xlen}')
        return xlen


###############################################################################
#######################     f-string formatting     ###########################
""" In general, f-string format specifiers are the same as '{}'.format() """

# Floating-point
#------------------
my_float = 2**(1/2)
print(f'{my_float}')
# 1.4142135623730951

print(f'{my_float: .3f}')
# 1.414


# String justification
#---------------------
b = 'baz'

# left
print(f'|{b:<12}|') # ---> |baz         |

# center
print(f'|{b:^12}|') # ---> |    baz     |

# right
print(f'|{b:>12}|') # ---> |         baz|

# fill empty space with char
print(f'|{b:*<12}|') # ---> |baz*********|
print(f'|{b:0^12}|') # ---> |0000baz00000|
print(f'|{b:^>12}|') # ---> |^^^^^^^^^baz|


# Pad string/num with zeros
# -------------------------
num = 7
print(f'{num:03}') # fill to 3 leading zeros
# ---> 007

print(f'{11:03}') # it properly interprets length of str
# ---> 011

###############################################################################
#######################        arg parsers          ###########################

# Fire
# ====
from fire import Fire

def neg(num):
    return -num

def poop(location='floor'):
    return f'poop on the {location}'

if __name__ == '__main__':
  # Fire()      # both neg and poop available at CLI, but can only use one :[
  # Fire(poop)  # 'fire_example.py airplane' ---> `poop on the airplane`


###############################################################################
#######################        pdfminer.six         ###########################

# NB: next to know API documentation; appears to be CLI-only?

pdf2txt.py a_pdf_file.pdf
# returns decent representation of extracted text



###############################################################################
#######################           tinydb            ###########################
# document-oriented database that stores data in dict-like form in
#  json

# Example usage (from docs)
# =============
from tinydb import TinyDB, Query
db = TinyDB('db.json') # akin to open('db.json', 'x')

# insert some data # NB: insert also returns id
db.insert({'type': 'apple', 'count': 7})
db.insert({'type': 'peach', 'count': 3})

# See all docs stored
db.all()
[{'count': 7, 'type': 'apple'}, {'count': 3, 'type': 'peach'}]

# Iterate over docs in db
for item in db:
    print(item)
{'count': 7, 'type': 'apple'}
{'count': 3, 'type': 'peach'}

# Query for specific documents
# ============================
Fruit = Query()
db.search(Fruit.type == 'peach')
[{'count': 3, 'type': 'peach'}]
db.search(Fruit.count > 5)
[{'count': 7, 'type': 'apple'}]

# To update certain fields
# ------------------------
db.update({'count': 10}, Fruit.type == 'apple')
db.all()
[{'count': 10, 'type': 'apple'}, {'count': 3, 'type': 'peach'}]

# To remove documents
# -------------------
db.remove(Fruit.count < 5)
db.all()
[{'count': 10, 'type': 'apple'}]

# And to remove ALL data
db.purge()
db.all()
[]

# To save db.json, simply
db.close()

###############################################################################
#######################             Misc            ###########################

# BIZARRE SHIT
# ============
float.as_integer_ratio(37, 103) # should return (37, 103) right??
(6471191697580907, 18014398509481984) # WRONG, returns this bullshit

# Works fine with simple ratios (0.5, 0.75, 0.25)
# not so good with primes...
float.as_integer_ratio(5/23)
(7832347178035645, 36028797018963968) # ... really?

# Handling decode errors in response
# ----------------------------------
page_response.decode('utf-8')
# ---> UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa0 in position 103997: invalid start byte
# JUST ADD 'ignore' to decode errors
page_response.decode('utf-u', 'ignore')
# allll gooood

# Get current date
#-------------------
import datetime
datetime.datetime.now().date()


# Handy dict
#------------------
class AttrDict(dict):
    # just a dict mutated/accessed by attribute instead index
    # NB: not pickleable
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Get $USER
# -------------------------
import os
user = os.environ['USER'] # easiest
# OR
import getpass
getpass.getuser()


# It turns out Python actually...has an array?
# --------------------------------------------
# And it's std lib...
from array import array

"""
class array(builtins.object)
 |  array(typecode [, initializer]) -> array
 |
 |  Return a new array whose items are restricted by typecode, and
 |  initialized from the optional initializer value, which must be a list,
 |  string or iterable over elements of the appropriate type.
 |
 |  Arrays represent basic values and behave very much like lists, except
 |  the type of objects stored in them is constrained. The type is specified
 |  at object creation time by using a type code, which is a single character.
 |  The following type codes are defined:
 |
 |      Type code   C Type             Minimum size in bytes
 |      'b'         signed integer     1
 |      'B'         unsigned integer   1
 |      'u'         Unicode character  2 (see note)
 |      'h'         signed integer     2
 |      'H'         unsigned integer   2
 |      'i'         signed integer     2
 |      'I'         unsigned integer   2
 |      'l'         signed integer     4
 |      'L'         unsigned integer   4
 |      'q'         signed integer     8 (see note)
 |      'Q'         unsigned integer   8 (see note)
 |      'f'         floating point     4
 |      'd'         floating point     8
 |
 |  NOTE: The 'u' typecode corresponds to Python's unicode character. On
 |  narrow builds this is 2-bytes on wide builds this is 4-bytes.
 |
 |  NOTE: The 'q' and 'Q' type codes are only available if the platform
 |  C compiler used to build Python supports 'long long', or, on Windows,
 |  '__int64'.

# Essentially all the same operations as normal collection data structures
#  like list. Just that it requires type specified in init.
#  Obviously faster ops for certain things, such as arr.count

"""

lst = [6, 12, 11, 5, 3, 15, 9, 1, 0, 14, 2, 7, 13, 10, 8, 4]
arr = array('l') # arr of signed ints (int32)
arr.fromlist(lst)
# --> array('l', [6, 12, 11, 5, 3, 15, 9, 1, 0, 14, 2, 7, 13, 10, 8, 4])


###############################################################################
#######################    namespace-like dict      ###########################
"""
Source:
  - SO post: https://stackoverflow.com/a/2597440
  - SO user: https://stackoverflow.com/users/95810/alex-martelli

If you have a dictionary `d` and want access behavior like namespace or
class attributes, such as my_dict.foo instead of my_dict['foo'],
simply do my_dict = Bunch(d)
"""
class Bunch:
    def __init__(self, dict_):
        self.__dict__.update(dict_)

class BunchInception: # assumes no dicts are elements of list
    def __init__(self, dict_ : dict):
        for k,v in dict_.items():
            if not isinstance(v, dict):
                self.__dict__[k] = v
            else:
                self.__dict__[k] = BunchInception(v)

class AttrDict(dict):
    """ simply a dict accessed/mutated by attribute instead of index
    WARNING: cannot be pickled like normal dict/object
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
