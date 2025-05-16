class Employee:
    raiseamount=1.05
    count=0
    def __init__(self,name,salary):
        self.name=name
        self.salary=salary
        Employee.count+=1


    def applyraise(self):
        self.salary=self.salary*self.raiseamount

    def getemail(self):
        return self.name+'@gmail.com'
    @classmethod
    def createobject(cls,string):
        first,salary=string.split('-')
        return cls(first,salary)


    @classmethod
    def employeecount(cls):
        return Employee.count
    def __repr__(self):
       return f"{self.name}-{self.getemail()}"
    def totalsalaries(*args):
        totalsalary=0
        for arg in args:
            totalsalary+=arg.salary
        return totalsalary




emp_str='ramesh-2200000'
ramesh=Employee.createobject(emp_str)
print(ramesh.salary)


#by inheritance we can inherit all the attributes of the parent class into its subclass

class developer(Employee):
    raiseamount=1.10
    def __init__(self,name,salary,pro_lang):
        super().__init__(name,salary)
        self.pro_lang=pro_lang



class manager(Employee):
    employeecount=0
    def __init__(self,name,salary,employees=None):
        super().__init__(name,salary)

        if employees is None:
            self.employees=[]
        else:
            self.employees=employees
            self.employeecount+=1

    def addemployee(self,emp):
        if emp not in self.employees:
            self.employees.append(emp)
            self.employeecount+=1
    def removeemployee(self,emp):
        if emp in self.employees:
            self.employees.remove(emp)
            self.employeecount-=1


    def printemployee(self):
        for emp in self.employees:
            print(emp)




raj=developer('raj',2200,'python')
print(ramesh.getemail())          #here we doesnt give any init method to the developer class still it
                                  #use the attributes of its parent class employee email as its attribute
print(help(developer))
print(raj.salary)
raj.applyraise()
print(raj.salary)
print(Employee.raiseamount)
print(developer.raiseamount)
print(developer.mro())
print(raj.pro_lang)

ramesh=developer('ramesh',2200,'java')
print(ramesh.pro_lang)
print(ramesh.name)
print(ramesh.salary)


manager1=manager('john',5000,[ramesh])
manager2=manager('alex',4000,[raj])

manager1.addemployee(raj)
manager1.__repr__()
manager1.printemployee()
print(manager1.employees)
print(manager1.employeecount)
print(manager2.employeecount)
manager1.removeemployee(raj)
print(manager1.employees)
print(manager1.employeecount)
print(manager1.salary)
print(Employee.totalsalaries(raj,ramesh,manager1,manager2))




