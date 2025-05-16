print("welcome to the tip calculator")
total_bill=float(input("what is yout total bill"))
persons_num=int(input("how many people split bill"))
percentage_tip=float(input("what percentage tip you will be giving 10 ,12,15"))
tip=total_bill*(percentage_tip/100)
tipper_person=tip/persons_num
print( "each persom should give",tipper_person)

