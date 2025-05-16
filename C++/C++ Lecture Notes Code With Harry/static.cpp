#include<iostream>
using namespace std;

 int product(int a,int b){

static int c=5;//this line executes only once next time this function is run the value of c will be retained
c=c+1;
return a*b+c;

}

int main(){
    int a,b;
    cin>>a>>b;
    cout<<product(a,b);

cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
cout<<product(a,b);
return 0;
}

//here static int c  means that the value of c is only used once by the function then
//it retains its value,when second time function initiats in main
//then the staic line not runs and the func gets the value as c