#include <iostream>

using namespace std;
// `cout` is part of std namespace
// if you wanted to use without "using namespace"
//  you would do std::cout<<"hello world"

int main()
{
    cout<<"Hello World"; //<< endl;
    return 0; // This exits the program
}


/*
Notes on C++:
`#include <iostream>` is a header


`using namespace std` allows us to use entities from the
std namespace
`cout` is part of std namespace
if you wanted to use without "using namespace"
you would do std::cout<<"hello world"



# Compiling
`g++ sum.cpp` will create an executable, `a.out`
`g++ sum.cpp -o sum` will name that executable 'sum'



# Debugging with GDB
# ==================
You need to compile with the -g flag. This builds the code with
debugging symbols, and enables it to work with GDB.

`g++ -g sum.cpp -o sum`
then:
`gdb sum`

# GDB commands
b line#: creates a breakpoint in the given line number.
         While debugging, the debugger stops at this break point.

n: executes the next line of code
r: runs the program until the break point
p variable_name: prints the value of a variable
q: exits the debugger

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from sum...done.
(gdb) b 5
'Breakpoint 1 at 0x8ed: file sum.cpp, line 5.
(gdb) r
Starting program: /home/evan/Cloud/Projects/AI-Sandbox/sandbox/ROS/sum

Breakpoint 1, main () at sum.cpp:7
7       int num_1 = 3;
(gdb) n
8       int num_2 = 4;
(gdb) p num 1
A syntax error in expression, near `1'.
(gdb) p num_1
$1 = 3
(gdb) q
A debugging session is active.

    Inferior 1 [process 9401] will be killed.

Quit anyway? (y or n) y


*/
