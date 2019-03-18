#!/usr/bin/env python
import os
import sys
import pdb
import code
import argparse
import traceback
from pprint import pprint
import numpy as np

# Snippets
# ========
file_path = os.path.abspath(os.path.dirname(__file__))
path_up1 = '/'.join(file_path.split('/')[:-1])
if not path_up1 in sys.path:
    sys.path.append(path_up1)

#code.interact(local=dict(locals(), **globals()))

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# parser
# ======
CLI = argparse.ArgumentParser()

# Add arguments here preceding any subparser

# subparsers
#subparsers = CLI.add_subparsers(dest='subcmd')
subparsers = None

# Subcommand decorator
def argp(*names_or_flags, **kwargs): return names_or_flags, kwargs

def subcmd(*parser_args):
    """ decorator used to define args to a subcommand """
    global subparsers
    if subparsers is None:
        subparsers = cli.add_subparsers(dest='subcmd')
    parent = subparsers
    def decorator(func):
        parser = parent.add_parser(func.__name__, description=func.__doc__)
        for args, kwargs in parser_args:
            parser.add_argument(*args, **kwargs)
        parser.set_defaults(func=func)
    return decorator

# parser args
CLI.add_argument('-f', '--foo', type=str, nargs='?', default=None, const='baz',
    help="sample 'optional with default' primary parser arg")

@subcmd(argp('-m', '--model', type=str, metavar='path', help='path/to/model'))
def pretrained(args):
    """ use pretrained model params """
    mpath = args.model
    if not os.path.exists(mpath):
        CLI.error('Path to model not found!')
    train.train(from_params=mpath)


def main():
    args = CLI.parse_args()
    subcommand = args.subcmd

    if args.foo is not None:
        foo = args.foo
        if foo != 'baz':
            print('foobar')
        else:
            print('baz')

    if subcommand is not None:
        CLI.func(args)

    # some WIP code that maybe raises an exception
    raise BaseException("oh no, exception!")
    return 0

if __name__ == "__main__":
    try:
        ret = main()
    except:
        traceback.print_exc()
        pdb.post_mortem()
    sys.exit(ret)
