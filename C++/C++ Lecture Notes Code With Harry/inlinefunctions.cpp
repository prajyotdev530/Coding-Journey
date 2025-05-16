#include<iostream>
using namespace std;
/*int product(int a,int b){

    return a*b;
}

int main(){
    int a,b;

cin>>a>>b;
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
return 0;*/
//running this code requires a lot of time for compilwe as everytime the product function is call
//by using inline functions we can directly replace the return vlaue with product(a,b)
//for making a inline function only write inline in the begginig of the function name
 inline int product(int a,int b){

    return a*b;
}

int main(){
    int a,b;

cin>>a>>b;
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
            return 0;
            }