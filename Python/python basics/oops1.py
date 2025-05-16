class User:
     def __init__(self,username):
             self.username=username
             self.followers=0
             self.following=0
     def follow(self,user):
         self.following+=1
         user.followers+=1

user_1=User("sarang")
user_2=User("prajyot")
user_1.follow(user_2)
print(user_2.followers)
print(user_1.following)
user_1.follow(user_2)
print(user_1.following)
print(user_1.following)



