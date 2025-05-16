#include <stdio.h>
int main()
{

    int x[10] = {1, 2, 3, 4};
    printf("%d", x[0]);
    scanf("%d", &x[4]);
    printf("%d", x[4]);
    return 0;
    //4 bytes are allocated to an integer in an array we can finf the address of the first element we can calculate of all
    //x+i*k
}