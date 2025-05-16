
using namespace std;

int search(int array[], int size)
{
    int b;
    for (int i = 0; i < size; i++)
    {
        b = array[i];
        if (b == 21)
        {
            return i;
        }
    }
    return -1;
}

int main()
{

    int array[] = {2, 3, 4, 5, 6, 7, 77, 7, 5, 4, 3, 32, 21, 1};
    int size = 14;
    cout<<search(array, 14);

    cout << sizeof(array);

    return 0;
}