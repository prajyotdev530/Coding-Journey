# Description: This file contains the function days_in_month(year, month) that returns the number of days in a month of a year.
import leapyear
year=int(input("enter the year:"))
month=int(input("enter the month:"))
def days_in_month(year,month):
    if leapyear.leap_year(year):
        if month==2:
            return 29
        elif month in [4,6,9,11]:
            return 30
        else:
            return 31
    else:
        if month==2:
            return 28
        elif month in [4,6,9,11]:
            return 30
        else:
            return 31
        
print(days_in_month(year,month))