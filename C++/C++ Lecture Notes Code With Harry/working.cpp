#include<iostream>
using namespace std;

float cheatfund(int investedmoney,float intrestrate=1.04)//here we write the default argument and the compulsory one id written first

{

    return investedmoney*intrestrate;
}
float cheatfund2(int,float);

int main(){

    int money,intrestrate;
    cin>>money>>intrestrate;
    cout<<"if you invested amount"<<money<<"then you get"<<cheatfund(money)<<endl;//here as i give only money hence it takes 
    //intrest rate default i.e. 1.04
    cheatfund2(money,intrestrate);
}
float cheatfund2(int money,float intrestrate){

    cin>>money>>intrestrate;
    cout<<"if you are viP"<<cheatfund(money,intrestrate);//but here we give both input of maoney as x and intrest rate as y so it take that value

    
    


}

