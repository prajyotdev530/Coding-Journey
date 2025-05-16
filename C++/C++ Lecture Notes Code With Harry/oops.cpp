#include<iostream>
#include<string>
using namespace std;


class binary{
    string s;
    public:
      void num(void);
      void checkbinary(void);
      void onescompliment(void);
      void display(void);

};
void binary::num(void){
    cout<<"enter the string";
    cin>>s;
}
void binary::checkbinary(void){

for(int i=0;i<s.length();i++){
if(s.at(i)!='0'&& s.at(i)!='1'){
    cout<<"the number is not binary";
    exit(0);
}
}
}
void binary::display(void){
    cout<<s;
}
void binary::onescompliment(void){

    for(int i=0;i<s.length();i++){

        if(s.at(i)=='0'){
            s.at(i)='1';
        }
        else{
            s.at(i)='0';
        }
    }
}
int main(){

binary b;
b.num();
b.checkbinary();
b.display();
b.onescompliment();
b.display();

return 0;
}