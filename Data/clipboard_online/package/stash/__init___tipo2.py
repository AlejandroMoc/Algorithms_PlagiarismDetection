import random as r

class SdD:
    def __init__(self, n):
        self.n = n
        self.lst = [r.randint(1, 100) for _ in range(self.n)]
    
    def gM(self):
        return sum(self.lst) / self.n

    def fLM(self):
        return min(self.lst)

    def fGM(self):
        return max(self.lst)

    def sL(self):
        self.lst.sort()

d = SdD(10)
print(d.lst)
print(d.gM())
print(d.fLM())
print(d.fGM())
d.sL()
print(d.lst)