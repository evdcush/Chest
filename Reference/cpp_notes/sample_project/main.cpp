#include "add.h"
using namespace std;


/*
Summary
=======
A linux makefile is a tool to compile one or more sources
in a single command and build the executable.

Common practice is to declare the class in a header and
cpp file, and then include the header in the main code for
getting that class.

So there are three files to this project:
 - main.cpp: the main code that we are going to build
 - add.h: the header file of the add class. It has a
          declaration of the class.
 - add.cpp: this file has the entire definition of the
            add class.

It's good practice to use the class name as the name of the
header and .cpp file. Here we create the add class so that
the name of the header is add.h and add.cpp

*/

int main()
{
    add obj;
    int result = obj.compute(43, 34);
    cout << "The Result:= " << result << endl;
    return 0;
}

/*

#-----------------------------------------------------------------------------#
#                                  makefile                                   #
#-----------------------------------------------------------------------------#

$ g++ add.cpp main.cpp -o main.o
$ ./main
The Result:= 77

Compiling multiple files
========================
The g++ command is convenient for compiling single-source code,
but if we want to compile several source codes, the
g++ command is inconvenent.

A Linux makefile is one way to compile multiple source codes
in a single command.

See the file `makefile` in this project directory.

Build the code via:
$ make
g++ -c main.cpp -o main.o
g++ -c add.cpp -o add.o
g++ main.o add.o -o main

$ ls
add.cpp add.h add.o main main.cpp main.o makefile

$ ./main
The Result:= 77

$ make clean
rm -f main.o add.o main

$ ls
makefile add.cpp main.cpp add.h


*/

/*
#-----------------------------------------------------------------------------#
#                                  CMakefile                                  #
#-----------------------------------------------------------------------------#

CMakefile
=========
CMakefile is another approach to building a C++ project.

It can build, test, and package software across multiple OS platforms.

See the `CMakeLists.txt` file. It basically sets the C++ flags and creates
an executable named main from the source code: add.cpp and main.cpp.
The list of CMake commands is available at cmake.org/documentation

After making the CMakeLists.txt, we have to create a folder for building
the project. You can choose any name for the folder.

Here, we use 'build' for that folder.

cd to the build folder, and run

`cmake ..`

This command parses CMakeLists.txt in the project path. The cmake command
can convert CMakeLists.txt to a makefile, and we can build the makefile
after that.

Basically, it automates the process of making the Linux makefile.


$ cmake ..
-- The C compiler identification is GNU 8.2.0
-- The CXX compiler identification is GNU 8.2.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /home/evan/Cloud/Projects/AI-Sandbox/sandbox/robotics/ros_for_nubs/cpp_project/build

$ make
Scanning dependencies of target main
[ 33%] Building CXX object CMakeFiles/main.dir/add.cpp.o
[ 66%] Building CXX object CMakeFiles/main.dir/main.cpp.o
[100%] Linking CXX executable main
[100%] Built target main

$ ls
CMakeFiles/ main Makefile cmake_install.cmake CMakeCache.txt

$ ./main
The Result:= 77

*/
