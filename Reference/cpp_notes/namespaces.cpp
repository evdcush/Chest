#include <iostream>

// NB: other cpp files do not use the std namespace
//     because cpp is new and I don't want to
//     make incorrect assumptions about what is available
//     at the global scope.

// So far, my understanding is that anything defined in the
// headers (the `#include`), like 'cout' and 'endl' from
// `iostream`, must be scoped in a namespace (std) for usage




/*

Summary
=======
To create a namespace, use the namespace keyword, followed by name
of the namespace.

Above, we defined two namespaces. You can see we defined the same
function in each namespace.

The namespaces are used to group a set of functions or classes
that perform a unique action.

We can access the members inside the namespace using the name of
the namespace followed by :: and the function name.

*/




using namespace std;

namespace robot {
    void process(void)
    {
        cout << "Processing by Robot" << endl;
    }
}

namespace machine {
    void process(void)
    {
        cout << "Processing by Machine" << endl;
    }
}

int main()
{
    robot::process();
    machine::process();
    /*
    $ ./out
    Processing by Robot
    Processing by Machine
    $
    */
}


