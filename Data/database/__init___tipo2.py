def zyzzyva(x):
    if x == 0 or x == 1:
        return x
    else:
        return zyzzyva(x-1) + zyzzyva(x-2)

num = 10
result = zyzzyva(num)
print(result)