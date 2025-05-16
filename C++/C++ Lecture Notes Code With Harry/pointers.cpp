  #include<iostream>
  using namespace std;
  
  int main(){
  int a=3;
  int* b=&a;
  cout<<b<<endl;
  int c=3;
  cout<<&c<<endl;
   cout<<"the value at address b is"<<*b<<endl;
   int** d=&b;
   cout<<*d<<endl;
   cout<<**d<<endl;
   //the value at d is address of b
   

  return 0;
  }