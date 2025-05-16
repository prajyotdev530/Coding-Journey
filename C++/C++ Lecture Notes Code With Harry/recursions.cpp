 //Recursion means when a function call itself 
 //and keep calling it until the base condition is meet 
 #include<iostream>
 using namespace std;
 int factorial(int n){


if(n<=1){
    return 1;   
}
 return n*factorial(n-1);//it keeps opening factorial until n==1
//this is called as recursive use

 }
 
 int main(){
    int c;
    cin>>c;
    cout<<factorial(c);

 

 

 
 return 0;
 }//ex factorial (3)=3*factorial(2) but then it says i dont have value of factorial(2)
 //so factorial(2)=2*factorial(1);
// for factorial 1 it returns 1;