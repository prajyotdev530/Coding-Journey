import random
letters_num=int(input("how many number of letters you want"))
symbols_num=int(input("how many number of symbols you want"))
numbers_num=int(input("how many number of numbers you want"))
number_num=int(numbers_num)
letters_num=int(letters_num)
symbols_num=int(symbols_num)
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
symbols = ['!','@','#','$','%','^','&','*','(',')','_','+']
numbers = ['0','1','2','3','4','5','6','7','8','9']
password=[]
for i in range(0,letters_num):
 password.append(random.choice(letters))
for i in range(0,symbols_num):
 password.append(random.choice(symbols))
for i in range(0,numbers_num):
    password.append(random.choice(numbers))
random.shuffle(password)
print(password)
password_string=""
for i in password:
  password_string=password_string+i
print(password_string)
logo="hello"
    

