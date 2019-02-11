#include <iostream>

using namespace std;

/*
Summary
=======
To handle an exception, we mainly use three keywords.
  - try: Inside the try block, we can write our code,
         which may raise an exception.

  - catch: If the try block raises an exception, the
           catch block catches the exception. We can
           decide what to do with that exception.

  - throw: We can throw an exception from the try block
           when the problem starts to show. If the throw
           statement is executed, it raises an exception
           caught by the catch block

General Structure for Exception Handling
----------------------------------------

try
{
    // Our code snippets
}
catch (Exception name)
{
    // Exception handling block
}
*/


int main()
{
    try
    {
        int num1 = 1;
        int num2 = 0;

        if (num2 == 0)
        {
            throw num1;
        }
    }
    catch(int e)
    {
        cout << "Exception found: " << e << endl;
    }
}
