

from turtle import Turtle
class Snake():
    def __init__(self):
        self.segment()
        self.createsnake()

    def createsnake(self):
        for i in range(0,3):
        tim=Turtle()
        tim.color("white")
        tim.shape("square")
        tim.penup()
        tim.goto(count,0)
        count-=20
        self.segment.append(tim)

    def move(self):
        for seg_num in range(len(segment)-1,0,-1):
         new_x=segment[seg_num-1].xcor()
         new_y=segment[seg_num-1].ycor()
         segment[seg_num].goto(new_x,new_y)

