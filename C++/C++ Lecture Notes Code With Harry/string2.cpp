#include<iostream>
using namespace std;

int main(){
 char name[100];//if we are giving array characters we must define its chaaracter number
  cin>>name;
  cout<<name;//here we can directly take array as a input
 cout<<endl;
  cout<<name[4];
  //if we give space in output it will ignore the characters after space ex if we give input hello world it will only print hello

cout<<strlen(name);//here if we give input hello world ad 4 of hello and 1 of null character


return 0;
}