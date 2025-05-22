import random as r

def g(x):
    if x == 0:
        return 0
    return x + g(x-1)

def h(lst):
    total = 0
    for i in lst:
        total += i
    return total

class J:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def get_name(self):
        return self.name

    def get_age(self):
        return self.age

def k():
    num = r.randint(1, 10)
    return num

n = k()
print(n)

m = g(5)
print(m)

o = h([1, 2, 3, 4, 5])
print(o)

p = J("Alice", 30)
print(p.get_name())
print(p.get_age())