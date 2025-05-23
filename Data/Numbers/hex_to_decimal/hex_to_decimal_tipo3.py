def hexadecimal_a_decimal(hex_str):
    try:
        numero_decimal = int(hex_str, 16)
        return numero_decimal
    except ValueError:
        return "Número hexadecimal inválido"

# Uso de ejemplo
hex_str = "1A"
decimal_number = hexadecimal_a_decimal(hex_str)
print(f"La representación decimal de hexadecimal {hex_str} es {decimal_number}")

# Casos límite y limitaciones:
# - Entrada: "G" (Número hexadecimal inválido)
# - Entrada: "" (Cadena vacía)

# Mejoras opcionales:
# - Añadir soporte para números hexadecimales de punto flotante