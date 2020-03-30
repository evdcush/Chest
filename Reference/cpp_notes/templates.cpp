#include <iostream>

using namespace std;


// NOTE: this content not in book
//  see: https://www.geeksforgeeks.org/template-specialization-c/
//       https://www.geeksforgeeks.org/templates-cpp/
//       https://www.geeksforgeeks.org/the-c-standard-template-library-stl/

/*

Summary
=======
Templates are a feature of C++.

The idea is to pass data type as a parameter so that we don't need to write
same code for different data types.

For example, consider a software company that needs sort() for different
data types. Rather than writing and maintaining the multiple codes,
we can write one sort() and pass data type as a parameter.

C++ hass two keywords to support templates:
 - 'template'
 - 'typename' (which can always be replaced by keyword 'class')

How do they work?
-----------------
Templates are expanded at compiler time.

This is like macros, but the difference is that the compiler does
type checking before template expansion (unlike macros).

The idea is:
source code contains ONLY function/class, but compiled code may contain
multiple copies of the same function/class.

*/




// a function template
template <typename T>
T myMax(T x, T y)
{
    return (x > y) ? x : y;
}

int main()
{
    cout << myMax<int>(3, 7) << endl;        // call myMax for int
    cout << myMax<double>(3.0, 7.0) << endl; // call myMax for double
    cout << myMax<char>('g', 'e') << end;;   // call myMax for char

    return 0;

    /*

    Under the hood
    --------------
    The compiler internally generates and adds code based on the type:

    `myMax<int>(3, 7)` :
    int myMax(int x, int y)
    {
        return (x > y) ? x : y;
    }

    `myMax<char>('g', 'e')` :
    char myMax(char x, char y)
    {
        return (x > y) ? x : y;
    }

    */
}


// Here is a template for Bubble Sort, using templates

template <class T>
void bubbleSort(T a[], int n){
    for (int i = 0; i < n; i++)
        for (int j = n - 1; i < j; j--)
            if (a[j] < a[j - 1])
                swap(a[j], a[j - 1]);
}
