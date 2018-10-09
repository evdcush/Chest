import argparse

P = argparse.ArgumentParser()

#==============================================================================
# add_args : the rules, the format
#==============================================================================


#------------------------------------------------------------------------------
#                           Optional args: '-arg'
#------------------------------------------------------------------------------
# Optional arguments
# ========================================

# "optional" args must be preceded by the '-' character
#   - any arguments following the optionals are considered "positional"
#----> The parser has no issue not receiving any vals for optionals
P.add_args('-f', '--foo', help='my_foo') # -f, --foo are optional, help positional

#  OKAY:
# >>> P.parse_args([])
# Namespace(foo=None)
# >>> P.parse_args(['-f', 'bar'])
# Namespace(foo='bar')

#------------------------------------------------------------------------------
#                           Required args: 'arg'
#------------------------------------------------------------------------------
# Required arguments
# ========================================
# Required args, on the other hand, are not preceded by '-', and must
# receive a value. They cannot follow optional args either.

P.add_args('-f', '--foo', help='my_foo')
P.add_args('poop')

#===== BAD
# >>> P.parse_args([])
#   usage: aparse.py [-h] [-f FOO] poop
#   aparse.py: error: the following arguments are required: poop
#   An exception has occurred, use %tb to see the full traceback.
#===== BAD
# >>> P.parse_args(['-f', 'bar'])
#   usage: aparse.py [-h] [-f FOO] poop
#   aparse.py: error: the following arguments are required: poop
#   An exception has occurred, use %tb to see the full traceback.

#==== OKAY:
# >>> P.parse_args(['diarrhea'])
# Namespace(poop='diarrhea')
#==== OKAY:
# >>> P.parse_args(['smells graet', '--foo', '80085'])
# Namespace(poop='smells graet', foo='80085')


#------------------------------------------------------------------------------
#                   Booleans in argparse:
#                      action='store_true'
#------------------------------------------------------------------------------
# Arparse does not really support type=bool,
#  BUT, you can get bool values directly, without type interp
#  through the action='store_true' flag
P.add_args('-f', '--foo', help='my_foo')
P.add_args('-d', '--use_dropout', action='store_true')

#==== OKAY:
# >>> vars(P.parse_args(['-d']))
#    {'foo': None, 'use_dropout': True}

# >>> vars(P.parse_args([]))
#    {'foo': None, 'use_dropout': False}

# >>> vars(P.parse_args(['-f', '123']))
#    {'foo': '123', 'use_dropout': False}

# >>> vars(P.parse_args(['--use_dropout']))
#    {'foo': None, 'use_dropout': True}
