name="hangman"
namelist=["h","a","n","g","m","a","n"]
name_list=["_","_","_","_","_","_","_"]
count=7
value=False
print(name_list)
for i in range(0,7):
   guess=input("Guess a letter: ")
   for i in range(0,7):
      if guess==name[i]:
         name_list[i]=guess
         print("correct guess")
         
        
   if namelist==name_list:
            print("you win")
            value=True
            break
   
   print(name_list)
import my_module
from my_module import logo,hello

print(logo)
print(hello)
   
   
   
   
   



