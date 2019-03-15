#!/usr/bin/env python
import sys
import pdb
import code
import argparse
import traceback
from pprint import pprint
import numpy as np


# snips
# =====
#code.interact(local=dict(locals(), **globals()))
class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# parser
# ======
CLI = argparse.ArgumentParser()
add_arg = CLI.add_argument


def main():
    # some WIP code that maybe raises an exception
    raise BaseException("oh no, exception!")
    return 0

if __name__ == "__main__":
    #args = CLI.parse_args()
    try:
        ret = main()
    except:
        traceback.print_exc()
        pdb.post_mortem()
    sys.exit(ret)
