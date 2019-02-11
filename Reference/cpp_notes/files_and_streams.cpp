#include <iostream>
#include <fstream>
#include <string>

/*
Summary
=======
fstream includes the following 3 data types:
- ofstream: stands for 'output file stream'. It is used
    to create a file and write data into it
- ifstream: Represents an input file stream. It is used to
    read data from files.
- fstream: Has both read and write capabilities.

(of course, {cin, cout} from iostream, and {string} from string)

*/


int main()
{
    std::ofstream out_file;
    std::string data = "Robot_ID=0";
    std::cout << "Write data:" << data << std::endl;
    out_file.open("Config.txt");
    out_file << data << std::endl;
    out_file.close();

    std::ifstream in_file;
    in_file.open("Config.txt");
    in_file >> data;

    std::cout << "Read data:" << data << std::endl;
    in_file.close();

    return 0;
}
