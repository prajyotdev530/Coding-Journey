from turtle import Turtle,Screen
tim=Turtle(shape="turtle")
screen=Screen()
tim.color(black)
color=["red","green","blue","black","purple","yellow","range"]
screen=Screen()
screen.setup(500,400)
user_bet=screen.textinput(title="make yopur bet",prompt="which turtle will win the game")
turtle_index=[-70,-40,-10,20,50,80]
for number in range(0,6):
    tim=Turtle(shape="turtle")
    tim.color(color[number])
    tim.penup()
    tim.goto(-230,turtle_index[number])

screen.exitonclick()
