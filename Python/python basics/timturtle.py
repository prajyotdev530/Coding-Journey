import random
colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "black", "white"]
from turtle import Turtle,Screen

timmy=Turtle()

timmy.shape("turtle")
timmy.color("red")
timmy.width(1)
timmy.forward(100)
timmy.left(90)
timmy.forward(90)
timmy.left(90)
timmy.forward(100)
timmy.left(90)
timmy.forward(100)

for i in range (3,10):
    for j in range(0,i):
        timmy.forward(100)
        timmy.left(360/i)
    timmy.color((random.choice(colors)))



    
        




screen=Screen()
screen.exitonclick()

