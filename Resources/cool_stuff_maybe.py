"""
Script to keep track of all the stuff that might be cool or useful I find
"""
import sys
import code
from fire import Fire


#sargs = sys.argv[1:]

#code.interact(local=dict(globals(), **locals()))

def neg(num):
    return -num

def poop(location='floor'):
    return f'poop on the {location}'


if __name__ == '__main__':
  #Fire()
  Fire(poop)




