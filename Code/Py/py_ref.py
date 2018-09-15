import os, sys, code, math, random
from functools import wraps

'''
# For debugging

## Creates a break point
code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
'''

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
    @wraps(f)
    def inspector(*args, **kwargs):
        print('\n Inspecting function: <{}>'.format(f.__name__))
        x = args
        y = kwargs
        z = f(*args, **kwargs)
        code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        return z
    return inspector

# EASILY ADD METHODS TO A CLASS INSTANCE
class Foo:
    def __init__(self):
        self.x = 42

foo = Foo()

def addto(instance):
    def decorator(f):
        import types
        f = types.MethodType(f, instance, instance.__class__)
        setattr(instance, f.func_name, f)
        return f
    return decorator

@addto(foo)
def print_x(self):
    print self.x

# foo.print_x() would print "42"


'''
#      ....           ....           ....           ....           ....           ....           ....
#     ||             ||             ||             ||             ||             ||             ||
# /"""l|\        /"""l|\        /"""l|\        /"""l|\        /"""l|\        /"""l|\        /"""l|\
#/_______\      /_______\      /_______\      /_______\      /_______\      /_______\      /_______\
#|  .-.  |------|  .-.  |------|  .-.  |------|  .-.  |------|  .-.  |------|  .-.  |------|  .-.  |------
# __|L|__| .--. |__|L|__| .--. |__|L|__| .--. |__|L|__| .--. |__|L|__| .--. |__|L|__| .--. |__|L|__| .--.
#_\  \\p__`o-o'__\  \\p__`o-o'__\  \\p__`o-o'__\  \\p__`o-o'__\  \\p__`o-o'__\  \\p__`o-o'__\  \\p__`o-o'_
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------



 _____    _____    _____    ____    _____   _____   _____   __   __   __
#/  _  \  |  __ \  /     \  |    \  |  ___| |  ___| /  ___\ |  |_|  | |  |     #
#|  _  |  |  __ <  |  <--<  |  |  | |  ___| |  __|  |  \_ \ |   _   | |  |     #
#\_/ \_/  |_____/  \_____/  |____/  |_____| |__|    \_____/ |__| |__| |__|     #
#    __    __  ___   __      __  __   __  __   _____   ____   _____   _____    #
# __|  |  |  |/  /  |  |    |  \/  | |  \|  | /     \ |  _ \ /     \ |  _  \   #
#|  |  |  |     <   |  |__  |      | |      | |  |  | |  __/ |  |  | |     /   #
#\_____/  |__|\__\  |_____| |_|\/|_| |__|\__| \_____/ |__|   \____ \ |__|__\   #
# _____    ______   __  __   ___  ___   __    __   ___  ___  ___  ___   _____  #
#/  ___>  |_    _| |  ||  |  \  \/  /  |  |/\|  |  \  \/  /  \  \/  /  |_   /  #
#\___  \    |  |   |  `'  |   \    /   |        |   >    <    \    /    /  /_  #
#<_____/    |__|    \____/     \__/     \__/\__/   /__/\__\    |__|    /_____| #

# ____   _____   _____  _____     __     _____    __  __  #
#|  _ \ |  _  \ /     \ |  __ \  |  |    |  ___| |  \/  | #
#|  __/ |     / |  |  | |  __ <  |  |__  |  ___| |      | #
#|__|   |__|__\ \_____/ |_____/  |_____| |_____| |_|\/|_| #

#     .     #
#       .   #
#   . ;.    #
#    .;     #
#     ;;.   #
#   ;.;;    #
#   ;;;;.   #
#   ;;;;;   #
#   ;;;;;   #
#   ;;;;;   #
#   ;;;;;   #
#   ;;;;;   # #
# ..;;;;;.. # #
#  ':::::'  # #
#    ':`    # #



    #    ;;;;;                                                                   ;;;;;     #
    #    ;;;;;      ____   _____   _____  _____     __     _____    __  __       ;;;;;     #
    #  ..;;;;;..   |  _ \ |  _  \ /     \ |  __ \  |  |    |  ___| |  \/  |    ..;;;;;..   #
    #   ':::::'    |  __/ |     / |  |  | |  __ <  |  |__  |  ___| |      |     ':::::'    #
    #     ':`      |__|   |__|__\ \_____/ |_____/  |_____| |_____| |_|\/|_|       ':`      #




#      ....           ....           ....           ....           ....           ....           ....
#     ||             ||             ||             ||             ||             ||             ||
# /"""l|\        /"""l|\        /"""l|\        /"""l|\        /"""l|\        /"""l|\        /"""l|\
#/_______\      /_______\      /_______\      /_______\      /_______\      /_______\      /_______\
#|  .-.  |------|  .-.  |------|  .-.  |------|  .-.  |------|  .-.  |------|  .-.  |------|  .-.  |------
# __|L|__| .--. |__|L|__| .--. |__|L|__| .--. |__|L|__| .--. |__|L|__| .--. |__|L|__| .--. |__|L|__| .--.
#_\  \\p__`o-o'__\  \\p__`o-o'__\  \\p__`o-o'__\  \\p__`o-o'__\  \\p__`o-o'__\  \\p__`o-o'__\  \\p__`o-o'_
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------


#   ;;;;;   #
#   ;;;;;   # #
# ..;;;;;.. # #
#  ':::::'  # #
#    ':`    # #
'''
