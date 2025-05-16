student_score=input("enter the score seperated by a space:").split()
print(student_score)
for n in range(len(student_score)):
    student_score[n]=int(student_score[n])
number=0
sum=0
print(student_score)
for n in student_score:
    number=number+1
    sum=sum+n
print(sum/number)