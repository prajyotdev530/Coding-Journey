def orignalfunction(origanl):
    def wrapperfunction(*args,**kwargs):
        for i in args:
            print(i)
        return orignal(*args,**kwargs)

@orignalfunction
def display():
    print("hello")

display()

