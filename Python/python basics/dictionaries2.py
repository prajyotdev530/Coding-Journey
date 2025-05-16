def f(**c):
    for x in c:
        print("the variable name is:",x,"the value of the variable isz:",c[x])
f(c1="a",c2="b",a="army",c=5.4)
l=[1,2,3,4,5]
l2=l
l2.append(6)
print(l)
print(l2)
l3=l.copy()
l3.append(7)
print(l)
print(l3)