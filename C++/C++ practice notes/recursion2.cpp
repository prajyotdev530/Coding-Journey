#include <iostream>
using namespace std;
int fibonacci(int a)
{
    if (a <= 2)
    {
        return 1;
    }
    return fibonacci(a - 1) + fibonacci(a - 2);
}

int main()
{
    int p;
    cin >> p;
    cout << "the value of" << p << "term is" << fibonacci(p);

    return 0;
}