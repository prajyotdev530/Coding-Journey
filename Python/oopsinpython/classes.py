class Person:
    raiseamount=1.04
    def __init__(self,username):
        self.username=username
        self.following=0
        self.followers=0
    def follow(self,user):
        self.following+=1
        user.followers+=1
user1=Person("john")
user2=Person("jim")
print(user1.following)
print(user2.following)
user1.follow(user2)
print(user1.following)
print(user2.followers)
user1.blocked=1
user2.blocked=1
print(user1.blocked)

class Employee:

     def __init__(self,first,last,pay):
            self.first=first
            self.pay=pay
            self.last=last
            self.email=first+last+"@gmail.com"


     def fullname(self):
        return '{} {}'.format(self.first,self.last)
     def applyraise(self):
         self.pay=int(self.pay*1.04)


user1=Employee("john","jim",5000000000000)
print(user1.email)
print(user1.fullname())
print(user1.pay)
user1.applyraise()
print(user1.pay)
Employee.raiseamount=1.05
user1.applyraise()
print(user1.pay)
print(user2.raiseamount)