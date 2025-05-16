def outer_func():
    message="hii"
    def inner_func():
        print(message)
    return inner_func

my_func=outer_func()  #here the innner function is copies into my_func and it is same and has
                      # accses to all the variables to
                      #to which the innerfunc as access
my_func()


#************************************************************************************************#
def newouter(msg):
    message=msg
    def newinner():
        print(message)
    return newinner

newmy=newouter("hello")
newmy()

