import random as rnd

class ProGra:
    def __init__(self):
        self.deFau = 0

    def roTa_te(self, list_a):
        for i in range(len(list_a)):
            rand_in = rnd.randint(i, len(list_a)-1)
            list_a[i], list_a[rand_in] = list_a[rand_in], list_a[i]
        return list_a

list_b = [1, 2, 3, 4, 5]
ins_tan_ce = ProGra()
print(ins_tan_ce.roTa_te(list_b))