#include <cstdio> // for printf, scanf


//----------------------------------------------------------------------------#
//                             Basic Data Types                               #
//----------------------------------------------------------------------------#
int    i; // '%d'  32-bit integer
long   l; // '%ld' 64-bit integer
char   c; // '%c'  Character type
float  f; // '%f'  32-bit real
double d; // '%lf' 64-bit real

// Reading data type
// -----------------
scanf("`format_specifier`", &val);

// eg, read char followed by doub
scanf("%c %lf", &c, &d); // ignore spacing at moment


// Printing data type
// ------------------
printf("`format_specifier`", val);

printf("%c %lf", c, d); // print char followed by double

char ch = 'j';
double d = 234.432;

// std::printf("%c %lf", ch, d); -->   `d 234.432000%`

// NB: You can use cin and cout instead of scanf and printf
//     however, if input 1m nums and print 1m lines, its faster to scanf/printf

/*---------------------------------------------------------------------------*
 *                                  Strings                                  *
 *---------------------------------------------------------------------------*/

/*
There are two types of strings you can use in C++:
    C-style strings
    cpp string class

*/


// C-style strings
// ===============
// Strings can be referred either as a character array or using
// a character pointer.

// strings as char arrays
// ----------------------
char str[4] = "Poo"; // One extra for string terminator
/*    OR     */
char str[4] = {'P', 'o', 'o', '\0'}; //  '\0' is string terminator

/*
   When strings are declared as char arrs, they are stored like
   other types of arrs in C.

   For example, if str[] is an auto variable, then the string is stored
   in a stack segment; if it's a global or static variable, the it is
   stored in a data segment, etc..
*/

// Strings using char pointers
// ---------------------------
/*
   Using char pointer strings can be stored in two ways:
   1) Read only string in a shared segment.
   2) Dynamically allocated in heap segment.
*/

char *str = "Poo";
// "Poo" is stored in shared read-only location, but pointer str
// is stored in a read-write mem. You can change str to point to something
// else, but it cannot change value at present str.







