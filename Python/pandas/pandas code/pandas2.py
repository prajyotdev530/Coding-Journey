import pandas as pd
df1=pd.read_csv('bios.csv')
df2=pd.read_csv('noc_regions.csv')

df1=df1.merge(df2,how='inner',on='NOC')
print(df1.loc[1:5,['name']])
