#include <iostream>
using namespace std;

int main() {
    int a = 42;
    int *ptr = &a;  // Pointer to `a`

    cout << "Value of a: " << a << endl;
    cout << "Address of a: " << &a << endl;
    cout << "Pointer (ptr) value (address of a): " << ptr << endl;
    cout << "Dereferencing ptr (*ptr): " << *ptr << endl;

    // Modifying the value using pointer
    *ptr = 100;
    cout << "Updated value of a: " << a << endl;

    return 0;
}