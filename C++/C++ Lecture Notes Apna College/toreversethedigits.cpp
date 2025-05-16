#include<iostream>
using namespace std;


int reverse(int a){
    int rem=0;
    int reverse=0;
    int power=1;
    while(a>0){
        rem=a%10;
       int b=a;
        while(b>0){
             
            rem=rem*10;
            b=b/10;
        }
        reverse=reverse+rem;
        a=a/10;

    
    }
    return reverse;
    



}

int main(){
    int a=21;
    cout<<reverse(31);







return 0;
}