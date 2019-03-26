#include <stdio.h>
typedef unsigned char byte;
void printBytes(const char * tag, unsigned char * bytes, int length);

/* CREDIT:
   github.com/pandasauce
   gist:
       https://gist.github.com/pandasauce/f8ea4b1776dd1cb9b8792ca907405abd
 */

int main()
{
    int original_variable = 100;
    int * ptr_to_variable = &original_variable;
    int resolved_variable = * ptr_to_variable;
    long wrong_type_deref = * ptr_to_variable;
    byte typedef_resolved_variable = * ptr_to_variable;
    char letter_resolved_variable = * ptr_to_variable;


    printBytes("original_variable", (unsigned char *) &original_variable, sizeof(original_variable));
    printBytes("ptr_to_variable", (unsigned char *) &ptr_to_variable, sizeof(ptr_to_variable));
    printBytes("resolved_variable", (unsigned char *) &resolved_variable, sizeof(resolved_variable));
    printBytes("* p", (unsigned char *) &(* ptr_to_variable), sizeof(* ptr_to_variable));
    printBytes("wrong_type_deref", (unsigned char *) &wrong_type_deref, sizeof(wrong_type_deref));
    printBytes("typedef_resolved_variable", (unsigned char *) &typedef_resolved_variable, sizeof(typedef_resolved_variable));
    printBytes("letter_resolved_variable", (unsigned char *) &letter_resolved_variable, sizeof(letter_resolved_variable));

    return 0;
}

void printBytes(const char * tag, unsigned char * bytes, int length)
{
    printf("%s: 0x", tag);
    int i;
    for (i = (length - 1); i >= 0; i--) {
        printf("%02x", bytes[i]);
    }
    printf("\n");
}
