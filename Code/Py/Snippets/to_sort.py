

"""

[&&&&&&&&&&&&&&&&&&] | [&&&&&&&&&&&&&&&&&&&&]


"""

def print_inplace(step, err, acc): #.format('STEP', 'ERROR', 'ACCURACY')
    #time.sleep(0.1)
    title = '{:<5}:  {:^7}   |  {:^7}'
    body  = '{:>5}:  {:.5f}   |   {:.4f}'
    s, e, a = step+1, float(err), float(acc)
    S, E, A = 'STEP', 'ERROR', 'ACCURACY'
    O = '\r{}'.format(body)
    ##status = O.format(S,E,A,s,e,a)
    if step == 0:
        print('\n' + title.format(S,E,A))
    sys.stdout.write(O.format(s,e,a))
    sys.stdout.flush()
    #time.sleep(0.05)
    #l = 40
    #e = round(l * float(err))
    #a = round(l * float(acc))
    #bar = '\r[{: <40}] | [{: <40}]'.format('@'*e, '@'*a)
    #sys.stdout.write(bar)
    #sys.stdout.flush()




###############################################################################
#
#
#               How to access parent class property

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
