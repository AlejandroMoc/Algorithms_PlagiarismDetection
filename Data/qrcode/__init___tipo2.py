import random as rnd

class GzrX:
    def __init__(self, n):
        self.n = n
        self.lst = [rnd.randint(1, 100) for _ in range(n)]

    def prnt_lst(self):
        for i in range(self.n):
            print(self.lst[i])

    def swp_elmnts(self, idx1, idx2):
        self.lst[idx1], self.lst[idx2] = self.lst[idx2], self.lst[idx1]

    def slct_srt(self):
        for i in range(self.n):
            min_idx = i
            for j in range(i+1, self.n):
                if self.lst[j] < self.lst[min_idx]:
                    min_idx = j
            self.swp_elmnts(i, min_idx)

n = 10
a = GzrX(n)
print("Lista antes de ordenar:")
a.prnt_lst()
a.slct_srt()
print("\nLista después de ordenar:")
a.prnt_lst()