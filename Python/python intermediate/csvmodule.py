import csv
with open('googleemployeesdata.csv','r') as employeedata:
    reader=csv.reader(employeedata)
    data=list(reader)
    for line in data:
        print(line)
    with open('newemployeesdata.csv','w') as newemployeedata:
        writter=csv.writer(newemployeedata,delimiter='\t')
        for line in data:
            writter.writerow(line)

    with open('newemployeesdata.csv','r') as newnewemployeedata:
        newreader=csv.reader(newnewemployeedata,delimiter='\t')
        newdata=list(newreader)
        for line in newdata:
            print(line)



#for using the dictionary reader we only need to add the DictReader instead of reader but while writting through
#the dictionary writter we need to do the following changes 

        with open('newfiledata.csv','w') as newfiles:

            new=csv.writer(newfiles,delimiter='\t')

            for line in newdata:
                new.writerow(line)
        with open('newfiledata.csv','r') as employeedata:
            newreader=csv.DictReader(employeedata,delimiter='\t')
            for line in newreader:
                print(line)






