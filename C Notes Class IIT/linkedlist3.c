#include<stdio.h>
#include<stdlib.h>
typedef struct list ELEMENT;
typedef ELEMENT *LIST;
typedef char DATA;

typedef struct list{
    DATA d;
    LIST next;;
    
};
void transversallist(LIST head){
    LIST ptr=head;
    while(ptr!=NULL){
        printf("%c",ptr->d);
        ptr=ptr->next;
    }
}
int main(){
LIST head;
head=(LIST)malloc(sizeof(ELEMENT));
LIST second;
second=(LIST)malloc(sizeof(ELEMENT));
head->d='a';
head->next=second;
second->d='b';
transversallist(head);
}