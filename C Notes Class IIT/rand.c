# include<stdio.h>
# include<stdlib.h>
int main(){
    
    for(int i=0; i<50; i++){
        double value = rand()%(2)+1;
        printf("%lf ", value);
    }
   return 0;
}