from turtle import Screen,Turtle
from snake import Snake
import time

screen=Screen()
screen.setup(600,600)
screen.bgcolor("black")
count=0
screen.title("My snake game")
screen.tracer(0)
segment=[]
snake=Snake()
game_is_on=True
while game_is_on:
     screen.update()
     time.sleep(0.1)
     snake.move()
  
    
     
     
         
         

         
     





screen.exitonclick()
