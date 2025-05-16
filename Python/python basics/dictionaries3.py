grades={"Harry":81,"Ron":78,"Hermione":99,"Draco":74,"Neville":70}
remark={}
for key,value in grades.items():
    if value>=91:
        remark[key]="outstanding"
    elif value>=81:
        remark[key]="Exceeds Exceptations"
    elif value>=71:
        remark[key]="Acceptable"
    else:
        remark[key]="Fail"

for i in grades:
    print(i,grades[i])

for i in remark:
    print(i,remark[i])