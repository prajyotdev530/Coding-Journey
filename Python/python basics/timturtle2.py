from turtle import Turtle,Screen
import random
tim=Turtle()
screen=Screen()

def forwardgo():
    tim.forward(10)
def backwardgo():
    tim.backward(10)
def leftgo():
    newheading=tim.heading()+10
    tim.setheading(newheading)
def rightgo():
    newheading=tim.heading-10
    tim.setheading(newheading)
def clearscreen():
     tim.home()
     tim.clear()
   

    

screen.listen()
screen.onkey(key="space",fun=forwardgo)
screen.onkey(key="w",fun=forwardgo)
screen.onkey(key="s",fun=backwardgo)
screen.onkey(key="d",fun=rightgo)
screen.onkey(key="a",fun=leftgo)
screen.onkey(key="c",fun=clearscreen)
screen.exitonclick()
