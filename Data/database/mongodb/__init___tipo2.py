def gafiru_zofusi(num):
    if num <= 1:
        return num
    else:
        return gafiru_zofusi(num-1) + gafiru_zofusi(num-2)

def zurfu_bilufu(max_num):
    for i in range(max_num):
        print(gafiru_zofusi(i))

zurfu_bilufu(10)