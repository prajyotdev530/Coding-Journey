#include<iostream>
using namespace std;


class animal{
char name;
int height;
int weight;
int id;
public:

void setdetails(int a,int b,int c);
void getdetails(void);
void feed();
};


void animal::setdetails(int a,int b,int c){
    height=a;
    weight=b;
    id=c;



}
void animal::getdetails(void){

    cout<<"the entered details are as following"<<endl<<"height"<<height<<endl<<"weight"<<weight<<"id"<<id;
}
void animal::feed(){
    cout<<"the feed required per month is"<<height*weight;
}



int main(){
animal dog;
dog.setdetails(3,4,5);
dog.getdetails();
dog.feed();

return 0;
}