def fozbaz(x):
    if x == 0 or x == 1:
        return x
    else:
        return fozbaz(x-1) + fozbaz(x-2)

def quxqux(num):
    for i in range(num):
        print(fozbaz(i))

quxqux(10)