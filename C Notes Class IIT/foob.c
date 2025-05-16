#include <stdio.h>

int foo(int n) {
    if (n <= 0) return 0;
    
    return n + foo(n/2) - foo(n-1);
}

int main(){
    printf("%d", foo(500));
    return 0;
}