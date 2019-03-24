# distutils: language = c++
# cython: language_level=3


# This code walkthrough from the excellent "Cython for NumPy users" tutorial:
# https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html


"""
### Cython file compiled into extension via:
$ cython cycompute.pyx  # creates cycompute.c

$ gcc -shared -pthread -fPIC -fwrapv -O2 -Wall \
-fno-strict-aliasing -I/home/evan/.pyenv/versions/3.7.2/include/python3.7m \
-o cycompute.so cycompute.c



### timeit tests run with following vars:

arr1 = np.random.uniform(0, 1000, size=(3000, 2000)).astype(np.intc)
arr2 = np.random.uniform(0, 1000, size=(3000, 2000)).astype(np.intc)
a = 4
b = 3
c = 9
fn_args = (arr1, arr2, a, b, c)

def npcompute(arr1, arr2, a, b, c):
    return np.clip(arr1, 2, 10) * a + arr2 * b + c

%timeit npcompute(*fn_args)
39.5 ms ± 585 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

import pycompute
%timeit pycompute.compute(*fn_args)
1min 20s ± 825 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

import cycompute
%timeit cycompute.compute(*fn_args)
# SEE RESULTS

"""

#-----------------------------------------------------------------------------#
#                        SLOW : dynamic typing, pyobj                         #
#                                                                             #
# 1min 9s ± 1.16 s per loop (mean ± std. dev. of 7 runs, 1 loop each)         #
#-----------------------------------------------------------------------------#
"""
Why slow?
=========
C code still does exactly what the Python interpreter does,
eg new objects being allocated for each num used
"""
import numpy as np

def clip_dynamic(a, vmin, vmax):
    return min(max(a, vmin), vmax)

def compute_dynamic(arr1, arr2, a, b, c):
    # implements: np.clip(arr1, 2, 10) * a + arr2 * b + c
    # arr1, arr2 are 2D
    assert arr1.shape == arr2.shape
    xmax = arr1.shape[0]
    ymax = arr1.shape[1]

    result = np.zeros((xmax, ymax), dtype=arr1.dtype)

    for x in range(xmax):
        for y in range(ymax):
            tmp = clip_dynamic(arr1[x,y], 2, 10)
            tmp = tmp*a + arr2[x,y] * b
            result[x,y] = tmp + c
    return result


#-----------------------------------------------------------------------------#
#                               FASTER : ctyped                               #
#                                                                             #
# 29.9 s ± 86.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)         #
#-----------------------------------------------------------------------------#
"""
Typed cython
============
While even standard CPython code will be faster in Cython,
to get any significant speedup in Cython, you must type all your
arguments and variables.

Why is it faster?
You cut huge swaths of runtime overhead whenever you reduce the
the stack. The normal CPython interpreter has an enormous C stack
for robust dynamic typing, with typechecks and testing up and down
the stack.

Typed vars mean the interpretter can make assumptions about what sort of
data is being passed to it's wrapped C code.
Less tests --> less overhead --> less time --> faster

Why still slower than numpy?
============================
Firstly, numpy is really fast.

What happens is that most of the time spent in this code is spent
in the following lines, and those lines are slower to execute
than in pure python:
    tmp = clip(arr1[x, y], 2, 10)
    tmp = tmp * a + arr2[x, y] * b
    result[x, y] = tmp + c

Why are these lines so much slower than in the CPython version?

arr1 and arr2 are still NumPy arrays, so pyobjs, and expect
Python integers as indices. In these lines, we pass C int values.

So every time Cython reaches this line, it has to convert all
the C integers to Python int objects.

Since this line is called very often, it outweighs the speedup
of the pure C loops that were created from the range() earlier.

Furthermore, `tmp * a + arr2[x, y] * b` returns a Python integer
and tmp is a C integer, so Cython has to do type conversions again!

In the end, those type conversions add up and make our computation
really slow.

But this problem can be solved easily using memoryviews.
--------------------------------------------------------

"""

#import numpy as np


# Py_ssize_t is a CPython typedef.
# It is used wherever CPython's C API functions accept or return
# a C-level integer that can be used for indexing Python sequences.
# Py_ssize_t in turn resolves to whatever the platform spelling
# is for the signed variant of the platform C's unsigned size_t type.
# It's a signed integer type, but its width (num bits) depends on your platform


# We now need to fix a datatype for our arrays.
# I've used the variable DTYPE for this.
# DTYPE is assigned to the usual NumPy runtime type info object.
DTYPE = np.intc # C type 'int'

# cdef means here that this function is a plain C function (so faster).
# To get all the benefits, we type the arguments and the return value.
cdef int clip_ctype(int a, int min_value, int max_value):
    return min(max(a, min_value), max_value)

def compute_ctype(arr1, arr2, int a, int b, int c):

    # The "cdef" keyword is also used within functions to type variables.
    # It can only be used at the top indentation level.
    cdef Py_ssize_t x_max = arr1.shape[0]
    cdef Py_ssize_t y_max = arr1.shape[1]

    assert arr1.shape == arr2.shape
    assert arr1.dtype == DTYPE
    assert arr2.dtype == DTYPE

    result = np.zeros((x_max, y_max), dtype=DTYPE)

    # It is very important to type ALL your variables. You do not get any
    # warnings if not, only much slower code (they are implicitly typed as
    # Python objects).
    # For the "tmp" variable, we want to use the same data type as is
    # stored in the array, so we use int because it correspond to np.intc.
    # NB: An important side-effect of this is that if "tmp" overflows its
    # datatype size, it will simply wrap around like in C, rather than raise
    # an error like in Python.

    cdef int tmp

    # Py_ssize_t is the proper C type for Python array indices.
    cdef Py_ssize_t x, y

    for x in range(x_max):
        for y in range(y_max):
            tmp = clip_ctype(arr1[x, y], 2, 10)
            tmp = tmp * a + arr2[x, y] * b
            result[x, y] = tmp + c
    return result


#-----------------------------------------------------------------------------#
#                        FASTER, PART II : memoryviews                        #
#                                                                             #
# 22.7 ms ± 71.5 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)      #
#-----------------------------------------------------------------------------#
"""
Bottlenecks
===========
There are still two bottlenecks that degrade the performance:
- array lookups
- assignments
as well as C/Python type conversions.

The []-operator still uses full Python operations, but we would
like to instead access the data buffer directly at C speed.

So we need to type the contents of the ndarray objects.
We do this with a memoryview.

memoryview
==========
In short, memoryviews are C structures that can hold a
pointer to the data of a NumPy array and all the necessary
buffer metadata to provide efficient and safe access:
dims, strides, item size, item type info, etc...

The also support slices, so they work even if the NumPy array
isn't contiguous in mem. They can be indexed by C ints,
thus allowing fast access to the numpy arr data.

Declaring a memoryview of ints:
cdef int [:] foo      # 1D memoryview
cdef int [:,:] foo    # 2D memoryview
cdef int [:,:,:] foo  # 3D memoryview
....                  # and so on

No data is copied from the NumPy arr to the memoryview in
our example. As the name implies, it is only a "view" of the
memory.

We can use the view `result_view` for efficient indexing and at
the end return the real NumPy array `result` that holds the data
that we operated on.

#==== Results
At 22.7ms, we are now approximately 3081 times faster than CPython,
and 4.5 times faster than NumPy

"""

#import numpy as np

DTYPE = np.intc

cdef int clip_memview(int a, int vmin, int vmax):
    return min(max(a, vmin), vmax)

def compute_memview(int[:,:] arr1, int[:,:] arr2, int a, int b, int c):
    # array_1.shape is now a C array, no it's not possible
    # to compare it simply by using == without a for-loop.
    # To be able to compare it to array_2.shape easily,
    # we convert them both to Python tuples.
    assert tuple(arr1.shape) == tuple(arr2.shape)

    cdef Py_ssize_t xmax = arr1.shape[0]
    cdef Py_ssize_t ymax = arr1.shape[1]

    # magic here
    result = np.zeros((xmax, ymax), dtype=DTYPE)
    cdef int[:,:] result_view = result

    cdef int tmp
    cdef Py_ssize_t i, j

    for i in range(xmax):
        for j in range(ymax):
            tmp = clip_memview(arr1[i,j], 2, 10)
            tmp = tmp * a + arr2[i,j] * b
            result_view[i,j] = tmp + c # result_view used for indexing
    return result # return true array


