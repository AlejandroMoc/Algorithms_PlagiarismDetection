def contar_bits_establecidos(num):
    count = 0
    while num:
        count += num & 1
        num >>= 1
    return count