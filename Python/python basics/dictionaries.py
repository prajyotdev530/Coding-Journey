def add(*args):
    sum=0
    for i in range(len(args)):
        sum=sum+i
    return sum
print(add(1,2,3,4,5))
list=[1,5,4,6]
list_sorted=sorted(list)
print(list_sorted)
print(list.count(0))
tel={"prajyot":100,"rahul":50}
print(tel["prajyot"])
print({x:x**2 for x in (2,4,6)})
tel["sarang"]=100
del tel["rahul"]
print(tel)
for key,value in tel.items():
    print(key,":",value)
student=dict(name="prajyot",age=21,college="IIT Kharagpur",package=40000000)
for key,value in student.items():
    print(key,":",value)
student["package"]=50000000
print(student)
print(student.get("name"))
print(student["name"])
student.pop("name")
print(student)
student.popitem()
print(student)

student=dict(name="prajyot",age=21,college="IIT Kharagpur",package=40000000)
student.setdefault("name","rahul")
print(student)
                                                             
print(type(employees))
