import os, sys, code, math, random
from functools import wraps


#     __  __  __   __   ___   ___   _  _    ___   _      ___                  #
#    |  \/  | \ \ / /  / __| |_ _| | \| |  / __| | |    | __|                 #
#    | |\/| |  \ V /   \__ \  | |  | .` | | (_ | | |__  | _|                  #
#    |_|  |_|   |_|    |___/ |___| |_|\_|  \___| |____| |___|                 #
#                                                                             #
#      ___   ___   ___     _    _____   ___   ___   _____                     #
#     / __| | _ \ | __|   /_\  |_   _| | __| / __| |_   _|                    #
#    | (_ | |   / | _|   / _ \   | |   | _|  \__ \   | |                      #
#     \___| |_|_\ |___| /_/ \_\  |_|   |___| |___/   |_|                      #
#                                                                             #
#      ___    ___   ___    ___    ___   __   __  ___   ___  __   __           #
#     |   \  |_ _| / __|  / __|  / _ \  \ \ / / | __| | _ \ \ \ / /           #
#     | |) |  | |  \__ \ | (__  | (_) |  \ V /  | _|  |   /  \ V /            #
#     |___/  |___| |___/  \___|  \___/    \_/   |___| |_|_\   |_|             #
#                                                                             #
#      ___   _  _    ___  __   __  _____   _  _    ___    _  _                #
#     |_ _| | \| |  | _ \ \ \ / / |_   _| | || |  / _ \  | \| |               #
#      | |  | .` |  |  _/  \ V /    | |   | __ | | (_) | | .` |               #
#     |___| |_|\_|  |_|     |_|     |_|   |_||_|  \___/  |_|\_|               #

# Creates a break point in computation
#  interupting code when it hits the line with the interact,
#  opening an interactive python shell
code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

# YES





#   __  __  __   __    ___     ___    ___   ___                            #
#  |  \/  | \ \ / /   |   \   / _ \  | _ \ | __|                           #
#  | |\/| |  \ V /    | |) | | (_) | |  _/ | _|                            #
#  |_|  |_|   |_|     |___/   \___/  |_|   |___|                           #
#                                                                          #
#   ___    ___    ___    ___    ___     _     _____    ___    ___   ___    #
#  |   \  | __|  / __|  / _ \  | _ \   /_\   |_   _|  / _ \  | _ \ / __|   #
#  | |) | | _|  | (__  | (_) | |   /  / _ \    | |   | (_) | |   / \__ \   #
#  |___/  |___|  \___|  \___/  |_|_\ /_/ \_\   |_|    \___/  |_|_\ |___/   #

"""
Why are ALL the decorator snippets online the same lame shit???

Either tutorials for babies, invariably using examples from official
documentation, or that tired old "celsius" crap from every tutorial
ever.

That, or ancient Py2 decorators for caching, HTTP, memo, etc.
Who uses those?

I don't. So I had to make my own slick decorators.

"""

# Nifty decorator to both comment and prevent running unfinished funcs
def TODO(f):
    """ Serves as a convenient, clear flag for developers and insures
        wrapee func will not be called """
    @wraps(f)
    def not_finished(*args, **kwargs):
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

#==============================================================================
#------------------------------------------------------------------------------
#                              OO/inheritance stuff
#------------------------------------------------------------------------------
#==============================================================================

