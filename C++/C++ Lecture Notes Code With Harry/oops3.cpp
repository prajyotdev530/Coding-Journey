#include<iostream>
using namespace std;



class employee{


    int price[100];
    int itemid[100];
   int counter;
    public:
     
    void setprice();
    void dispayprice();
    void initcounter(){counter=0;}

};
void employee::setprice(){
for(int i=0;i<=3;i++){
    cout<<"enter the id of number of counter"<<counter+1;
    cin>>itemid[i];

  
    cout<<"enter the price of item";
    cin>>price[i];  
counter++;
}
}
void employee::dispayprice(){
    for(int i=0;i<=3;i++){
        cout<<"the item id is"<<itemid[i]<<"and the price is"<<price[i];
    }
}





int main(){
employee prajyot;
prajyot.initcounter();
prajyot.setprice();
prajyot.dispayprice();


return 0;
}