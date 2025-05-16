#include<stdio.h>
#include<stdlib.h>
#include<time.h>
void printarray(int arr1[],int arr2[],int arr3[],int n){
for(int i=0;i<n;i++){
    if(i==n-1){
        printf("| %d |",arr1[i]);
        break;
    }
    else{printf("| %d  ",arr1[i]);}


};

printf("\n");
for(int i=0;i<n;i++){
    if(i==n-1){
        printf("| %d |",arr2[i]);
        break;
    }
    else{printf("| %d  ",arr2[i]);}
}

printf("\n");
for(int i=0;i<n;i++){
    if(i==n-1){
        printf("| %d |",arr3[i]);
        break;
    }
    else{printf("| %d  ",arr3[i]);}
}

printf("\n");
printf("---------------");
printf("\n");


}
int takeinput(int arr1[],int arr2[],int arr3[],int n){
int x;
printf("enter the row between 1 to 3");
scanf("%d",&x);
int y;
printf("enter the coloum you want to enter between 1 to 3");
scanf("%d",&y);

int z;
printf("enter the input 0 or1");
scanf("%d",&z);
if(x==1){
    if(arr1[y-1]==0||arr1[y-1]==1){
        printf("already filled");
        return 0;
    }
    else{arr1[y-1]=z;}
}
if(x==2){
    if(arr2[y-1]==0||arr2[y-1]==1){
        printf("already filled");
        return 0;
    }
    else{arr2[y-1]=z;}
}
if(x==3){
    if(arr3[y-1]==0||arr3[y-1]==1){
        printf("already filled");
        return 0;
    }
    else{arr3[y-1]=z;}
}


}
void cominput(int arr1[],int arr2[],int arr3[],int n){
    srand(time(NULL));
    int x=rand()%3+1;
    int y=rand()%3+1;
    int z=rand()%1+1;
    if(x==1){
        if(arr1[y-1]==0||arr1[y-1]==1){
            printf("already filled");
            
        }
        else{arr1[y-1]=z;}
    }
    if(x==2){
        if(arr2[y-1]==0||arr2[y-1]==1){
            printf("already filled");
            
        }
        else{arr2[y-1]=z;}
    }
    if(x==3){
        if(arr3[y-1]==0||arr3[y-1]==1){
            printf("already filled");
           
        }
        else{arr3[y-1]=z;}
    }
}

int main(){
int r1[3]={3,3,3};
int r2[3]={3,3,3};
int r3[3]={3,3,3};
printarray(r1,r2,r3,3);
takeinput(r1,r2,r3,3);
printarray(r1,r2,r3,3);

cominput(r1,r2,r3,3);
printarray(r1,r2,r3,3);
takeinput(r1,r2,r3,3);
printarray(r1,r2,r3,3);

cominput(r1,r2,r3,3);


return 0;





}



