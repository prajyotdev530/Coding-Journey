def outer_function(orignal_function):
    def innerfunction():
        return orignal_function()
    return innerfunction

def display():
    print("hello")

myfunction=outer_function(display)
myfunction()

#we can do the same task by addiing @outer_function to the starting of the display function
def newouter(oringnalfunction):
    def newinner():
        return oringnalfunction()
    return newinner

@newouter
def newdisplay():
    print("hell")

newdisplay()



def decoratorfunction(orignal_function):
    def deccorator_inner(*args, **kwargs):
        return orignal_function(*args, **kwargs)
    return deccorator_inner

@decoratorfunction
def dis():
    print("heloworld")
dis()

