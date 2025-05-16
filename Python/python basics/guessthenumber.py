import random 
number=int(random.randint(1,100))
level=input("what level do you want easy or hard")
if level=="easy":
  count=10
elif level=="hard":
  count=5
def guessnumber():
   global count
   if count==0:
       return "you lose"
       print(number)
      
   print(f"you have{count}chances left to guess")
   guess=int(input("enter the guess number"))
   if guess>number:
      print("too high")
      count=count-1
      guessnumber()
      
   elif guess<number:
      print("too low")
      count=count-1
      guessnumber()
      
   else:
      return ("you won you guess the correct number")
    
      
print(guessnumber())

      



