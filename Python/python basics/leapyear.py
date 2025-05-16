
def leap_year(year):
    """it tells whethe a year is leap year or not."""
    if(year%4==0):
        if(year%100==0):
            if(year%400==0):
                return True
            else:
                 return False
        else:
            return False
    else:
        return False

# Compare this snippet from lovecalculator.py: