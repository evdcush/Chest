import os, sys
import glob

#==============================================================================
# Get list of files
#==============================================================================
# SEE MORE METHODS: https://stackoverflow.com/a/41447012/6880404


# Everything in that dir
# ========================================
files_from_os = os.listdir()
'''
print(files_from_os)
# just every file a that dir
['work.txt', 'template.rst', 'notes_on_thing.rst', 'notes_on_thing2.rst',
 'cheat_sheet.rst', 'read_script.py', 'file_rw_and_io_stuff.py',]
'''

# Matching files of same type or something in common
# ========================================
rst_files = []
for file in glob.glob("*.rst"):
    rst_files.append(file)
# Or simply:
rst_files = [f for f in glob.glob("*.rst")]
'''
print(rst_files)
# Now only files with .rst extension:
['template.rst', 'notes_on_thing.rst', 'notes_on_thing2.rst', 'cheat_sheet.rst',]
'''

#==============================================================================
# READ/WRITE file
#==============================================================================

# Options for open
#========= ===============================================================
# Character Meaning
# --------- ---------------------------------------------------------------
# 'r'       open for reading (default)
# 'w'       open for writing, truncating the file first
# 'x'       create a new file and open it for writing
# 'a'       open for writing, appending to the end of the file if it exists
# 'b'       binary mode
# 't'       text mode (default)
# '+'       open a disk file for updating (reading and writing)



# Read a file, line by line
#--------------------------
lines = open('file.txt').read().split('\n')

# OR
with open('file.txt') as f:
    lines = f.read().split("\n")

# OR
with open('file.txt', -r) as f:
    lines = f.readlines()

# OR
lines = []
with open('file.txt', -r) as f:
    for line in f.readlines():
        lines.append(line)

# OR
f = open('file.txt') # Open file on read mode
lines = f.read().split("\n") # Create a list containing all lines
f.close() # Close file


# Write to a file
#--------------------------
f = open('file_to_write', 'x')
for line in lines:
    f.write(line + '\n')
f.close()

# or
with open('file_to_write', 'w') as f:
    f.write('file contents')



#==============================================================================
# STD:IN, STD:OUT examples
#  (from ACM)
#==============================================================================

# input() only takes in the very first line

n = int(input())
a = list()
c = list()
for i in range(n):
    c.append(list(map(int, input().split())))


#N = int(lines[0])
#arr = [int(line) for lin in lines[1:-1]]
#k = int(lines[-1])
#print('lines: {}, N: {}, arr: {}, k: {}'.format(lines, N, arr, k))

a, b = map(int, input().split())
n, a, b = map(int, raw_input().split())


''' # sample input
4 9
PP.-PPPS-S.S
PSP-PPSP-.S.
.S.-S..P-SS.
P.S-P.PP-PSP
'''
n, k = map(int, input().split())
plain = list()
for i in range(n):
    plain.append(list(input()))



''' # sample input
3 18
4 4 4 2 2 2
'''
n, w = map(int, input().split())
a = list(map(int, input().split()))


line = sys.stdin.readline()
lines = sys.stdin.readlines()# ['1\n', '2\n', ...]
