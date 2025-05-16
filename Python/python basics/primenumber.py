def prime_check(num):
   
    for i in range(2,num):
        if num%i==0:
            print("the number is not a prime number")
            break
        if(i==num-1):
            print("the number is a prime number")

number=input("enter the number you want to check")
number_int=int(number)
prime_check(number_int)