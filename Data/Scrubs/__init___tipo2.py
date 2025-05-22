def snafoo_ker():
    lorp = 0
    for boop in range(1, 6):
        lorp += boop
    return lorp

class Bizzle:
    def __init__(self, baz):
        self.baz = baz

    def frobnicate(self, quux):
        return self.baz + quux

snafu = Bizzle(10)
result = snafu.frobnicate(5)
print(result + snafoo_ker())