class employee:
    raiseamount=1.04
    def __init__(self,name,age,salary):
        self.name=name
        self.increment=0
        self.salary=salary
        self.age=age
        self.email=name+'@gmail.com'
    def applyraise(self):
        self.salary=self.salary*self.raiseamount
    @classmethod
    def setraiseamount(cls,amount):
        employee.raiseamount=amount
    @classmethod
    def fromstring(cls,emp_str):
        first,age,salary=emp_str.split('-')
        return cls(first,age,salary)
    @staticmethod
    def workday(day):
        if day.weekday()==5 or day.weekday()==6:
            return False
        return True





ramesh=employee('ramesh',22,100000000)
print(ramesh.increment)
print(ramesh.salary)
ramesh.raiseamount=1.05
print(employee.raiseamount)
print(ramesh.salary)
employee.raiseamount=1.05
print(employee.raiseamount)
print(ramesh.salary)
ramesh.applyraise()
print(ramesh.salary)
employee.setraiseamount(1.06)
print(ramesh.email)

emp_1='suresh-21-100000'
emp_2='rajesh-22-10000000'

first,last,pay=emp_1.split('-')
emp_1_str=employee(first,last,pay)
emp_2_str=employee.fromstring(emp_2)
print(emp_1_str.name)

print(emp_2_str.name)
import datetime
my_day=datetime.date(2025,5,9)
print(ramesh.workday(my_day))




