import os, sys, code, math, random
from functools import wraps

'''
# For debugging

## Creates a break point
code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
'''

# Nifty decorator to both comment and prevent running unfinished funcs
def TODO(f):
    @wraps(f)
    def not_implemented(*args, **kwargs):
        print('\n  FUNCTION NOT IMPLEMENTED: <{}>'.format(f.__name__))
    return not_implemented

def INSPECT(f):
    @wraps(f)
    def inspector(*args, **kwargs):
        print('\n Inspecting function: <{}>'.format(f.__name__))
        x = args
        y = kwargs
        z = f(*args, **kwargs)
        code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    return inspector


