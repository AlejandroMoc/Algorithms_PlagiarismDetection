def gzzz(n):
    if n <= 1:
        return n
    else:
        return gzzz(n-1) + gzzz(n-2)

def xxyy(num):
    sequence = []
    for i in range(num):
        sequence.append(gzzz(i))
    return sequence

output = xxyy(10)
print(output)