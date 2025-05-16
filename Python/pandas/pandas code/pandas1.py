import pandas as pd
import numpy as np
dataframe=pd.DataFrame([[1,2,3],[1,2,3],[5,6,7]],columns=['a','b','c'],index=['x','y','z'])
print(dataframe)
import csv

print(dataframe.head())
print(dataframe.head(1))  #it will show the first row of the dataframe if it has 2
# it will show the first 2 rows simpilarly for the tail function
print(dataframe.info())
#with open('coffee.csv','r') as file:
   # reader=csv.reader(file)
    #for row in reader:
       # print(row)
coffee=pd.read_csv('coffee.csv')
print(coffee)
print(coffee.mean)
#we can read the parquet or he excel file by doing results=pd.read_parquet('file name')
#and we can also convert the file by reults=to_parquet('')
print(coffee.sample(10))#this will give the random sample of 10 datas
print(coffee.loc[1])   #loc helps in giving the rows data
print(coffee.loc[[1,2,3]]) #this will give us the first second and the third row
print(coffee.iloc[1])
print(coffee.loc[5:12,['Day','Units Sold']])   #it will grab 5 to 12 rows and the specified columns
print(coffee.loc[:,['Day']])
#for using the iloc we specified the column index
print(coffee.iloc[:,[0,2]])  #this will grab all rows and the column of 0,2
coffee.loc[1,'Units Sold']=6
print(coffee)
print(coffee.loc[1,'Units Sold'])
print(coffee.loc[[1,2,3],'Units Sold'])
print(coffee['Day'])
print(coffee.iloc[:,[0,1]])
coffee=coffee.sort_values('Units Sold',ascending=False)
print(coffee)
print(coffee.sort_values(['Units Sold','Coffee Type'],ascending=[False,True]))   #thsi will first sort the units sold in the descending order then it will sort the coffee tyoe t]in the alphabetical
#order in ascending way
#we can also add the column using
coffee['prices']=4.99
print(coffee)
coffee['new_price']=np.where(coffee['Coffee Type']=='Espresso',3.99,5.99)
coffee['revenue']=coffee['prices']*coffee['new_price']
for index,row in coffee.iterrows():
    print(index)
    print(row)
    print("\n\n\n")
bios=pd.read_csv('bios.csv')
print(bios)
print(bios.head())
print(bios.info())
print(bios.describe())
print(bios.loc[:,'born_region'])
print(bios.iloc[:,[0,1]])
print(bios.loc[bios['height_cm']>140])
print(bios[(bios['height_cm']>215) & (bios['born_country']=='USA')])
print(bios['name'].str.contains('USA'))
print(bios[bios['name'].str.contains('Keith',case=False)])
print(bios.query('born_country=="USA" & born_city=="Seattle"'))
print(coffee[coffee['revenue']>10])
#bios=bios['name'].str.split(' ').str[0]
print(bios)
print(bios['height_cm'].apply(lambda x:'short' if x<165 else ('average' if x<185 else 'tall')))
biosregions=pd.read_csv('noc_regions.csv')
print(bios)
print(biosregions)
bios_new=pd.merge(bios,biosregions,left_on='born_country',right_on='NOC',how='left')
print(bios_new.head())
#we can form a new object or table using
USA=bios[bios['born_country']=='USA']
print(USA)
GBR=bios[bios['born_country']=='GBR']
#WE CAN CONCATINATE THE STRINGS USING
USAGBR=pd.concat([USA,GBR])
print(USAGBR)
from sklearn.datasets import load_iris
iris_dataset = load_iris()
bios.merge(biosregions, left_on='born_country',right_on)
