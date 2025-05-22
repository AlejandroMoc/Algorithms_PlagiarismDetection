import random as r

def lskd_fsd():
    hsd = r.randint(1, 10)
    for i in range(hsd):
        print(f"Valor {i+1}: {r.random()}")

class HsdKlsd:
    def __init__(self, name):
        self.name = name
    
    def hsd_mtd(self):
        print(f"Hola, {self.name}!")

lskd_fsd()
obj = HsdKlsd("Mundo")
obj.hsd_mtd()