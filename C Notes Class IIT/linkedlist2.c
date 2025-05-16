#include<stdio.h>
#include<stdlib.h>


struct list{
    int data;
    struct list *next;
};


int main(){

struct list * head;
struct list * second;
struct list * third;
head=(struct list*)malloc(sizeof(struct list));
second=(struct list*)malloc(sizeof(struct list));
third=(struct list*)malloc(sizeof(struct list));
head->data=10;
head->next=second;
second->data=20;
second->next=third;
third->data=30;
third->next=NULL;
printf("%d",head->data);

}