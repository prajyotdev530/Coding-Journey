#include<iostream>
using namespace std;
int main(){
    int age;
    cin>>age;
switch(age){

    case 18:
    cout<<"you are 18"<<endl;
    break;
    case 2:
    cout<<"you are 2"<<endl;
    break;
    case 3:
    cout<<"you are 3"<<endl;
    break;
    default:
    cout<<"no special cases"<<endl;
    break;
}
//here if we doesnt include break them it
 //will print every cout and if we put break
 // and age 18 the it will print 18 and take exit from code


}
