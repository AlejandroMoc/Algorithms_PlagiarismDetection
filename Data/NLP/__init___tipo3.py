def suma_cadena(cadena):
    suma = 0
    for caracter in cadena:
        if caracter.isdigit():
            suma += int(caracter)
    return suma

def multiplicar_cadena(cadena):
    producto = 1
    for caracter in cadena:
        if caracter.isdigit():
            producto *= int(caracter)
    return producto

cadena = "abc123def456ghi"
suma = suma_cadena(cadena)
producto = multiplicar_cadena(cadena)

print("La suma de los dígitos en la cadena es:", suma)
print("El producto de los dígitos en la cadena es:", producto)