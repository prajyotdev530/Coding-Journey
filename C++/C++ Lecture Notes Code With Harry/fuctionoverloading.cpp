#include<iostream>
using namespace std;



int sum(int a,int b){
cout<<"using function with two argumernts"<<endl;
    return a+b;
}
int sum(int x,int y,int z){
cout<<"using function with three arguments"<<endl;
return x+y+z;

}

int main(){
    int p,q,r;
    cin>>p>>q>>r;
    cout<<sum(p,q)<<endl;
    
    cout<<sum(p,q,r);
    //here we had two functions of same name so the 
    //compiler itself gives the right value with right numbers of variables 
    //to the write function5
    



return 0;
}