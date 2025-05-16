#include<iostream>
using namespace std;


int main(){
int a;

cin>>a;
for(int i=2;i<(a-1);){

    if(a%i==0){
        cout<<"number is not prime";
        break;
    }
    else if(i==(a-2)){
        cout<<"number is prime";
        break;
    }
    else{
        i++;
        
    }

}



return 0;
}
/*int main(){

int a;
cin>>a;
bool isprime=true;
for(int i=2;i<a;){


if(a%i==0){


isprime = false;
cout<<"the number is not prime";
break;}
 else{
 i++;}
}
if(isprime==true){cout<<"the number is prime";
}
return 0;
}*/








