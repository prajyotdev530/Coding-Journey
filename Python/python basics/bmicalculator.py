weight=float(input("enter your weight in kg:"))
height=float(input("enter your height in m:"))
bmi=weight/height**2
bmi_as_int=int(bmi)
if bmi<18.5:
    print("you are underweight")
else:
    (print("you are overweight"))