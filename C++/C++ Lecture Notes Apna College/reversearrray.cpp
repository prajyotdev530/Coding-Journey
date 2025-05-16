#include <iostream>
using namespace std;

void swap(int array[100])
{
    int a = 4;

    for (int i = 0; i < a; i++)
    {

        swap(array[i], array[a]);
        a--;
    }
}
int main()
{
    int array[100] = {1, 2, 3, 4, 5};
    swap(array);
    for (int i = 0; i < 5; i++)
    {
        cout<< array[i];
    }

    return 0;
}