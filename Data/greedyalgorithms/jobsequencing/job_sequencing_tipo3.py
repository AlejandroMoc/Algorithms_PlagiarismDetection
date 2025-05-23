class Trabajo:
    def __init__(self, identificador, plazo, beneficio):
        self.identificador = identificador
        self.plazo = plazo
        self.beneficio = beneficio

def comparar(trabajo):
    return trabajo.beneficio

def minimo_numero(num1, num2):
    return num2 if num1 > num2 else num1

def secuencia_trabajos(trabajos, n):
    # Ordenar trabajos por beneficio en orden descendente
    trabajos.sort(key=comparar, reverse=True)

    # Lista para almacenar el resultado (secuencia de identificadores de trabajo)
    resultado = [-1] * n
    # Lista booleana para mantener el seguimiento de los espacios de tiempo ocupados
    espacio = [False] * n

    # Iterar a través de todos los trabajos
    for i in range(n):
        # Encontrar un espacio libre para este trabajo, revisando desde el último espacio posible
        for j in range(minimo_numero(n, trabajos[i].plazo) - 1, -1, -1):
            if not espacio[j]:  # Si el espacio está libre
                resultado[j] = i  # Asignar este trabajo al espacio
                espacio[j] = True  # Marcar el espacio como ocupado
                break

    # Obtener la secuencia de identificadores de trabajo para el beneficio máximo
    secuencia_trabajo = [trabajos[resultado[i]].identificador for i in range(n) if espacio[i]]
    return secuencia_trabajo

# Ejemplo
if __name__ == "__main__":
    n = 4
    trabajos = [
        Trabajo('a', 4, 20),
        Trabajo('b', 1, 10),
        Trabajo('c', 1, 40),
        Trabajo('d', 1, 30)
    ]

    # Mostrar la secuencia de trabajos que maximiza el beneficio
    print("La siguiente es la secuencia de trabajos que maximiza el beneficio:")
    print(" ".join(secuencia_trabajos(trabajos, n)))