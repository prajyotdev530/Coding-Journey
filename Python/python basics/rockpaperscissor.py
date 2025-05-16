import random
yourchoice=input("enter your choice 0 for rock, 1 for paper, 2 for scissor: ")
if yourchoice=="0":
    print("you choosse rock")
elif yourchoice==1:
    print=("you choose paer")
else:
    print("you choose scissor")

computerchoice=random.randint(0,2)
if computerchoice==0:
    print("computer choose rock")
elif computerchoice==1:
    print("computer choose paper")
else:
    print("computer choose scissor")
if(int(yourchoice)==computerchoice):
        print("draw")
elif(int(yourchoice)==0 and computerchoice==2 or int(yourchoice)==1 and computerchoice==0 or int(yourchoice)==2 and computerchoice==1):
        print("you win")
else:
    print("you lose")