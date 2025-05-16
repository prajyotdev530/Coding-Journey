

#f=open("usa_data.json") it will throw error as the file name is incorrect

try:
    f=open("usa_data.json")
except Exception:
    print("sorry this file doesnot exist")  #this will catch all the rror in the try block but we need a specific type of error to exceppt so we d the folowing thoing


try:
    f=open("usad_ata2.json")
    #var=badvar
except FileNotFoundError as e:
    print(e)   #her eit willl give error as we only except the file not found error so the var error is shown
except Exception as e:
    print(e)
else:                  #if any error form the except or any is not hit then the else is hot and executed
    print(f.read())
    f.close()
finally:     #this runs always if it hit error or not it wil run
        print("finally is printed ")

try:
    f=open("googleemployeesdata.csv")
    if f.name=="googleemployeesdata.csv":          #raising error
        raise Exception
except Exception:
    print("sorry this file doesnot exist")
list=[x**2 for x in range(10)]
print(list)
