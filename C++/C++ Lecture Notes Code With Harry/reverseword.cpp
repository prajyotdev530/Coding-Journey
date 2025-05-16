#include <iostream>
#include<string>
using namespace std;

class reverse
{

    string s;

public:
    void takesstring();
    void hii();
    int a;

}
void reverse::takesstring(){

cout<<"enter the text you want to reverse";
cin>>s;
}
void reverse::hii(){
  
int a=s.length();
for(int i=0;i<s.length();i++){

    s.at[a]=s.at[i];
    a--;
}

}
int main(){

    reverse prajyot;
    

}