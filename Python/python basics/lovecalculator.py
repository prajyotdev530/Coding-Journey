your_name=input("enter your name:")
therir_name=input("enter their name:")
true_count=your_name.count("t")+your_name.count("r")+your_name.count("u")+your_name.count("e")+therir_name.count("t")+therir_name.count("r")+therir_name.count("u")+therir_name.count("e")
love_count=your_name.count("l")+your_name.count("o")+your_name.count("v")+your_name.count("e")+therir_name.count("l")+therir_name.count("o")+therir_name.count("v")+therir_name.count("e")
love_score=true_count*10+love_count
print("your percentage love is",love_score)
if love_score<10 or love_score>90:
    print("you go together like coke and mentos")       
elif love_score>40 and love_score<50:
    print("you are alright together")
else:
    print(f"your love score is {love_score}")