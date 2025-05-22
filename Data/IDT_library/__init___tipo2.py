import itertools as it

def xzv(n):
    return it.islice(it.count(), n)

def ymn():
    for i in xzv(10):
        print(i)

ymn()