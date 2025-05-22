def binario_a_decimal(cadena_binaria):
    try:
        numero_decimal = int(cadena_binaria, 2)
        return numero_decimal
    except ValueError:
        return "Número binario inválido"

# Uso de ejemplo
cadena_binaria = "1010"
numero_decimal = binario_a_decimal(cadena_binaria)
print(f"La representación decimal del binario {cadena_binaria} es {numero_decimal}")

# Casos límite y limitaciones:
# - Entrada: "2" (Número binario inválido)
# - Entrada: "" (Cadena vacía)

# Mejoras opcionales:
# - Añadir soporte para números binarios de punto flotante