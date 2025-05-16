#include<iostream>
using namespace std;

int counter;
class farmer{
int id[100];
int price[100];
static int counter;
public:
void setprice();
void displayprice();
};
void farmer::setprice(){
    for(int i=0;i<3;i++){


        cout<<"enter the farmer id"<<"of farmer no"<<counter<<endl;
         cin>>id[i];
         cout<<"enter the prize"<<endl;
         cin>>price[i];
         counter++;
    }

}
void farmer::displayprice(){
 
;
    for(int i=0;i<3;i++){
 cout<<"the id of farmer no"<<counter<<"is"<<id[i];
 cout<<"the prize of his crop  is"<<price[i];    }
counter++;
}


int farmer::counter;
int main(){
farmer krish,prajyot,sarang;
krish.setprice();
krish.displayprice();

prajyot.setprice();
prajyot.displayprice();

sarang.setprice();
sarang.displayprice();



return 0;
}