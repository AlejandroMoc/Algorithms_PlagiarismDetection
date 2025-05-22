def es_potencia_de_dos(num):
    return num > 0 and (num & (num - 1)) == 0  # Una potencia de dos tiene solo un bit establecido