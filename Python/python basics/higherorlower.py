'''import random
items={"rheana":350,"jupiter":400,"jet":500,"nid":600,"comb":3}
score=0
def guess(items):
    game=True
    global score
    list_size=5
    a=random.choice(list(items.keys()))
    b=random.choice(list(items.keys()))
    while a==b:
        b=random.choice(list(items.keys()))
    while game==True:
        print("who do you think has more followers")
        print(f"A){a}")
        print(f"B){b}")
        guess=input("a or b")
        if guess=="a" and items[a]>items[b]:
            score=score+1
            b=a
            del items[b]
        elif guess=="a" and items[a]<items[b]:
            game=False
        elif guess=="b" and items[b]>items[a]:
            score=score+1
            del items[a]
        elif guess=="b" and items[a]>items[b]:
            game=False
        a=random.choice(list(items.keys()))
    return score

guess(items)
print(score)'''

import random

items = {"rheana": 350, "jupiter": 400, "jet": 500, "nid": 600, "comb": 3}
score = 0

def guess_game(items):
    global score
    game = True

    while len(items) > 1 and game:  # Ensure at least two items exist
        a, b = random.sample(list(items.keys()), 2)  # Pick two unique items
        
        print("\nWho do you think has more followers?")
        print(f"A) {a} ")  # You can remove numbers for real game experience
        print(f"B) {b} ")

        user_input = input("Choose 'a' or 'b': ").strip().lower()

        if user_input == "a" and items[a] > items[b]:
            score += 1
            del items[b]  # Remove the loser
        elif user_input == "b" and items[b] > items[a]:
            score += 1
            del items[a]  # Remove the loser
        else:
            game = False  # End game on wrong answer

    print(f"Game over! Your final score: {score}")
    return score

guess_game(items)







        


        
