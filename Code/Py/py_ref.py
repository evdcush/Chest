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


#==============================================================================
#------------------------------------------------------------------------------
#                              OO/inheritance stuff
#------------------------------------------------------------------------------
#==============================================================================

import math

# try out ABC stuff too

class A(object):
    def __init__(self, num1, num2):
        self.num1 = num1
        self.num2 = num2
        self.math_fun_ret = 0

    def foo(self):
        print('foo')

    def bar(self):
        print('bar')

    def frog(self):
        print('frog and pi')
        return math.pi

    def do_some_math(self, num):
        math_stuff = math.sqrt(num)
        self.math_fun_ret = math_stuff
        print('A.do_some_math({}) = math.sqrt({}) = {}'.format(num, num, math_stuff))
        return math_stuff



    def __call__(self):
        self.foo()
        self.bar()
        abc_sum = self.a + self.b + self.c
        print('abc_sum = {}'.format(abc_sum))

class B(A):
    def __init__(self, a, b, c):
        #super(B, self).__init__(a,b,c)
        super().__init__(a,b,c) # same thing


    def foo(self):
        print('cat') # override?

    def bar(self):
        '''
        if B.bar() only calls super(B).bar(), then there is no need to
        define bar in B!
        b.bar() will automatically call super(B).bar() if undefined
        '''
        super(B, self).bar() # is this necessary? NO!

    # note: frog has not been defined in B

    def __call__(self):
        '''
        In: b = B(5,6,7)
        In: b()
        Out:
          cat
          bar
          poop
          now invoke super call
          cat
          bar
          abc_sum = 18
        '''
        self.foo()
        self.bar()
        print('poop')
        print('now invoke super call')
        super(B, self).__call__()

class C(B):
    # note no init at all
    '''
    instantiating c = C() with no init args will throw error
    c = C(a,b,c) is OKAY

    c.frog() is inherited from A through B, as you would expect
    '''


    def __call__(self):
        '''
        only calls B.foo(), B.bar(), since C.__call__ overrides B, but
        inherits B.foo(), B.bar() as self.foo, self.bar
        '''
        self.foo()
        self.bar()


class D(A):
    '''
    D tests whether you can simply change just one thing within a child class,
    but have it otherwise be the same as parent.

    All I want to do is add 2 to A.do_some_math(num)
     - check ret value is A.do_some_math(num) + 1
     - check if self.math_fun_ret is the result of A.do_some_math(num)
    '''

    def do_some_math(self, num):
        '''
        In: d.do_some_math(7)
        A.do_some_math(7) = math.sqrt(7) = 2.6457513110645907
        Out: 3.6457513110645907

        and self.math_fun_ret == 2.6457513110645907

        so if there are any self assignments within the parent class fun
        those will be made regardless of mutations in overridden child fun

        so if you want b.math_fun_ret to be result of b.do_some_math(num)
        rather than A.do_some_math(num), you need to explictly assign in B

        You may want to copy-paste the parent function in child function, but
        beware the risk of substitution errors.
        '''
        parent_ret = super(D, self).do_some_math(num)
        return parent_ret + 1







#==============================================================================
#------------------------------------------------------------------------------
#                              *args, **kwargs stuff
#------------------------------------------------------------------------------
#==============================================================================




def prod_with_fixArgs(a,b,c,d,e, square=False):
    """ sample product function with fixed args kwarg
    Note that a,b,c,d,e can be in list or tuple, if passed to the function
    with the "unpack" or "splat" operator *. See sample outputs below

    ''' # Sample outputs
    In: prod_with_fixArgs(1,2,3,4,5)
    Out: 120
    In: prod_with_fixArgs(1,2,3,4,5, True)
      squared product
    Out: 14400

    In: lst = [1,2,3,4,5]
    In: prod_with_fixArgs(*lst)
    Out: 120
    In: lst = [1,2,3,4,5, True]
    In: prod_with_fixArgs(*lst)
      squared product
    Out: 14400
    '''
    """
    prod = a*b*c*d*e
    if square:
        print('squared product')
        prod *= prod
    return prod

def prod_with_lstArgs(lst, square=False):
    """ sample product function with list arg
    '''
    In: lst = [1,2,3,4,5]
    In: prod_with_lstArgs(lst)
    Out: 120
    In: lstb = [1,2,3,4,5,True]
    In: prod_with_lstArgs(lstb)
    Out: 120
    '''
    """
    prod = 1
    for i in lst:
        prod *= i
    if square:
        print('squared product')
        prod *= prod
    return prod

def prod_with_varArgs(*args, square=False):
    """ prod function with variable number of args
    '''
    In: prod_with_varArgs(1,2,3,4,5,6)
    Out: 720
    In: prod_with_varArgs(1,2,3,4,5,6,square=True)
      squared product
    Out: 518400

    In: lst = [1,2,3,4,5,6]
    In: prod_with_lstArgs(lst)
    Out: [1,2,3,4,5,6] # *args assumed to be ints, not lists
    '''
    """
    prod = 1
    for i in args:
        prod *= i
    if square:
        print('squared product')
        prod *= prod
    return prod

def prod_with_varKwargs(**kwargs):
    """ prod function with kwargs
    Think of kwargs as a dict, where the keyword args are key:value tuples

    so prod_with_varKwargs(a=1,b=2,c=3,d=4,e=5, square=False)
     **kwargs is {a:1, b:2, c:3, d=4, e=5, square=False}

    '''
    In: prod_with_varKwargs(a=1, b=2, c=3, d=4)
    Out: 24
    In: prod_with_varKwargs(a=1, b=2, c=3, d=4, square=False)
    Out: 24
    In: prod_with_varKwargs(a=1, b=2, c=3, d=4, square=True)
      'squared product'
    Out: 576
    '''
    """
    prod = 1
    for key,val in kwargs.items():
        # key is always of type 'str'
        if key != 'square':
            prod *= val
    #if kwargs['square']: # returns a KeyError if kwargs['square'] is not defined!
    if 'square' in kwargs:
        if kwargs['square']:
            print('squared product')
            prod *= prod
    return prod

def prod_with_varArgsAndKwargs(*args, **kwargs):
    """ prod function with args and kwargs

    Here the items to prod will only be in args, whereas the product
    modifiers (like square) are in kwargs


    '''
    In: prod_with_varArgsAndKwargs(1,2,3,4,5):
    Out: 120
    In: prod_with_varArgsAndKwargs(1,2,3,4,5,square=True):
      squared product
    Out: 14400
    In: prod_with_varArgsAndKwargs(1,2,3,4,5,square=True, sub1=False):
      squared product
    Out: 14400
    In: prod_with_varArgsAndKwargs(1,2,3,4,5,square=True, sub1=True):
      squared product
      sub 1
    Out: 14399
    '''
    """
    prod = 1
    for i in args:
        prod *= i
    for key, val in kwargs.items():
        if key == 'square':
            if val:
                print('squared product')
                prod *= prod
        if key == 'sub1':
            if val:
                print('sub 1')
                prod -= 1
    return prod

def prod_with_fixArgs_varArgsAndKwargs(a,b,c, *args, **kwargs):
    """ prod function with fixed args a,b,c, variable args and kwargs
    note that any args past the first 3 are part of *args,
      and any keyword args are part of kwargs

    Args:
        a,b,c (int): prod ints
        *args: nums to subtract from prod
        **kwargs:
            square (bool): square prod
            sub1 (bool): subtract 1 from prod

    '''
    In: prod_with_fixArgs_varArgsAndKwargs(1,2,3):
    Out: 6
    In: prod_with_fixArgs_varArgsAndKwargs(1,2,3,4):
      *args to sub from prod
    Out: 2  # (6 - 4) = 2
    In: prod_with_fixArgs_varArgsAndKwargs(1,2,3,4,5):
      *args to sub from prod
    Out: -14 # (6 - 20) = -14

    In: prod_with_fixArgs_varArgsAndKwargs(1,2,3,4,5,square=True, sub1=True):
      *args to sub from prod
      squared product
      sub 1
    Out: 196 #(6 - 20)**2 - 1 = 196
    '''
    """
    prod = a*b*c
    if args:
        prod_diff = 1
        for i in args:
            prod_diff *= i
        print('*args to sub from prod')
        prod -= prod_diff
    #for i in args: prod_diff *= i
    for key, val in kwargs.items():
        if key == 'square':
            if val:
                print('squared product')
                prod *= prod
        if key == 'sub1':
            if val:
                print('sub 1')
                prod -= 1
    return prod

def prod_with_fixArgsAndKwargs_varArgsAndKwargs(a,b,c, *args, square=False,**kwargs):
    """ prod function with fixed args a,b,c, variable args and kwargs
    note that any args past the first 3 are part of *args,
      and any keyword args are part of kwargs

    Args:
        a,b,c (int): prod ints
        *args: nums to subtract from prod
        square (bool): if square: prod *= prod
        **kwargs:
            sub1 (bool): subtract 1 from prod

    '''
    In: prod_with_fixArgsAndKwargs_varArgsAndKwargs(1,2,3):
    Out: 6
    In: prod_with_fixArgsAndKwargs_varArgsAndKwargs(1,2,3,square=True):
      squared product
    Out: 36
    '''
    """
    prod = a*b*c
    if args:
        prod_diff = 1
        for i in args:
            prod_diff *= i
        print('*args to sub from prod')
        prod -= prod_diff

    if square:
        print('squared product')
        prod *= prod
    for key, val in kwargs.items():
        if key == 'sub1':
            if val:
                print('sub 1')
                prod -= 1
    return prod



class Foo(object):
    def __init__(self, num, pnum=15, **kwargs):
        self.num = num
        self.pnum = pnum
    def __call__(self, *args):
        foo_out = self.num + self.pnum
        formatted_out = '{} + {} = {}'.format(self.num, self.pnum, foo_out)
        print('FOO: num + plus_num = ' + formatted_out)

class Bar(object):
    def __init__(self, num, snum=3, **kwargs):
        self.num = num
        self.snum = snum
    def __call__(self, bar_call_numArg):
        bar_out = self.num - self.snum
        bar_out += bar_call_numArg
        formatted_out = '({} - {}) + {} = {}'.format(self.num, self.snum, bar_call_numArg, bar_out)
        print('BAR: (num - sub_num) + call_num = ' + formatted_out)

class FooBar(object):
    """ Composite of Foo and Bar, see how kwargs are passed to constituent
    attributes

    kwargs must be passed as **kwargs, not kwargs!
    # This bad!
    self.foo = Foo(num, kwargs)
    foo()
    @ num_sum 'TypeError: unsupported operand type(s) for +: 'int' and 'dict''
    - so kwargs being passed as a dict

    # **kwargs must be in Foo and Bar too!
    foobar1 = FooBar(num) # this okay!
    foobar2 = FooBar(num, sub_num=6) # this bad! sub_num undefined in Foo!
    # SOLUTION: put **kwargs in Foo and Bar!, then foobar2 works!
    """
    def __init__(self, num, **kwargs):
        #self.foo = Foo(num, kwargs) # bad!
        self.foo = Foo(num, **kwargs) # good!
        self.bar = Bar(num, **kwargs)

    def __call__(self, *args):
        self.foo(*args) # foo needs *args in __call__!
        self.bar(*args)












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


#                       ______                                                 #
#                      /\     \                                                #
#                     /  \     \                                               #
#                    /    \_____\                                              #
#                   _\    / ____/_                                             #
#                  /\ \  / /\     \                                            #
#                 /  \ \/_/  \     \                                           #
#                /    \__/    \_____\                                          #
#               _\    /  \    / ____/_                                         #
#              /\ \  /    \  / /\     \                                        #
#             /  \ \/_____/\/_/  \     \                                       #
#            /    \_____\    /    \_____\                                      #
#           _\    /     /    \    / ____/_                                     #
#          /\ \  /     /      \  / /\     \                                    #
#         /  \ \/_____/        \/_/  \     \                                   #
#        /    \_____\            /    \_____\                                  #
#       _\    /     /            \    / ____/_                                 #
#      /\ \  /     /              \  / /\     \                                #
#     /  \ \/_____/                \/_/  \     \                               #
#    /    \_____\                    /    \_____\                              #
#   _\    /     /_  ______  ______  _\____/ ____/_                             #
#  /\ \  /     /  \/\     \/\     \/\     \/\     \                            #
# /  \ \/_____/    \ \     \ \     \ \     \ \     \                           #
#/    \_____\ \_____\ \_____\ \_____\ \_____\ \_____\                          #
#\    /     / /     / /     / /     / /     / /     /                          #
# \  /     / /     / /     / /     / /     / /     /                           #
#  \/_____/\/_____/\/_____/\/_____/\/_____/\/_____/                            #
