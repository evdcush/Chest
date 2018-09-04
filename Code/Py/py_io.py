import math
import os
import random
import re
import sys



# input() only takes in the very first line

n = int(input())
a = list()
c = list()
for i in range(n):
    c.append(list(map(int, input().split())))

lines = open('input.txt').read().split('\n')
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


for line in X:
    f.write(line + '\n')

f.close()



##### OPENING FILE,READING LINE BY LINE
lines = open('file.txt').read().split("\n")

with open('file.txt') as fp:
    lines = fp.read().split("\n")

fp = open('file.txt') # Open file on read mode
lines = fp.read().split("\n") # Create a list containing all lines
fp.close() # Close file
line = sys.stdin.readline()


lines = sys.stdin.readlines()# ['1\n', '2\n', ...]
for line in lines:
  #print(line)
  print(int(line.strip()))

#### WRITING TO FILE

with open('file_to_write', 'w') as f:
    f.write('file contents')
