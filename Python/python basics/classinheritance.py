class Animal:
    def __init__(self):
        self.color="red"
    def breathe(self):
        print("breathe")
class fish(Animal):
    def __init__(self):
        super().__init__()
    def swim(self):
        print("swim")
tommy=fish()
print(tommy.color)
tommy.breathe()
tommy.swim()