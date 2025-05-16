import random
cards=[11,2,3,4,5,6,7,8,9,10,10,10,10]
my_cards=[]
computer_cards=[]
def hitsmy(my_cards):
      my_cards.append(random.choice(cards))
def hitscomputer(computer_cards):
    computer_cards.append(random.choice(cards))
def my_cards_sum(my_cards):
     summy=0
     for i in my_cards:
          summy+=i
     return summy  
def computer_cards_sum(computer_cards):
     sumcomputer=0
     for i in computer_cards:
          sumcomputer+=i
     return sumcomputer         
def game(my_cards,computer_cards):
    hitsmy(my_cards)
    hitsmy(my_cards)
    hitscomputer(computer_cards)
    hitscomputer(computer_cards)
    print(f"your cards are{my_cards}computer cards are{computer_cards[0]}")
    while computer_cards_sum(computer_cards)<17:
         hitscomputer(computer_cards)
         if computer_cards_sum(computer_cards)>21:
            return "you win"
    choice=input("do you want to hit or stand")
    if choice=="hit":
         hitsmy(my_cards) 
         if my_cards_sum(my_cards)>21:
            print(f"your cards are{my_cards}computer cards are{computer_cards}")
            return ("you lose")
         else:
           if computer_cards_sum(computer_cards)>my_cards_sum(my_cards):
              print(f"your cards are{my_cards}computer cards are{computer_cards}")
              return "you lose"
        
           else:
             print(f"your cards are{my_cards}computer cards are{computer_cards}")
             return("you win")
    elif choice=="stand":
         if computer_cards_sum(computer_cards)>my_cards_sum(my_cards):
              print(f"your cards are{my_cards}computer cards are{computer_cards}")
              return "you lose"
        
         else:
             print(f"your cards are{my_cards}computer cards are{computer_cards}")
             return("you win")
         
    print(f"your cards are{my_cards}computer cards are{computer_cards[0]}")
         
print(game(my_cards,computer_cards))
