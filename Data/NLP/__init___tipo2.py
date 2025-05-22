import random as rnd

def jklpqr(st):
    if len(st) <= 1:
        return [st]
    res = []
    for i, letter in enumerate(st):
        for perm in jklpqr(st[:i] + st[i+1:]):
            res += [letter + perm]
    return res

input_str = "abc"
result = jklpqr(input_str)
for perm in result:
    print(perm)