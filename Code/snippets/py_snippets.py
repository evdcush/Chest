import os
import sys






#########################################################
#-----> os pathing funcs


# returns path of CALLER, not callee script containing
#  the actual os.getcwd()
os.getcwd()



### ***********************************************************
# HOW TO GET DIRECTORY OF CURRENT FILE, NO FUCKED BULLSHIT BASED ON CALLER PWD

# So, if these lines were located in the following file:
#  "foo/bar/baz/poo.py"
import os

file_dir = os.path.abspath(os.path.dirname(__file__))

# file_dir ------>  "foo/bar/baz"
#  NO BULLSHIT WITH SYS.PATH
#  NO BULLSHIT WITH OS.PATH
#  NO BULLSHHHIT WITHH SUBPROCESS
