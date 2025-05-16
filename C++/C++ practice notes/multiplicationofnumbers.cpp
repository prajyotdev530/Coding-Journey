#include<iostream>
using namespace std;


/*int main(){

int result=1;
for(int i=1;i<=10;i++){
    result= result * i;
}
cout<<result;

return 0;
}



*/
//this can also be used
/*#include<iostream>
 using namespace std;
 int factorial(int n){
 
if(n==1){
    return 1;
}
 return n*factorial(n-1);


 }
 
 int main(){
    int c;
    cin>>c;
    cout<<factorial(c);

 return 0;
 }*/






int main(){
    int i;
cin>>i;
int result=1;
do{
result=result*i;
i--;
if(i==1){
    break;
}

}
while(i<=10);


    cout<<result;
}