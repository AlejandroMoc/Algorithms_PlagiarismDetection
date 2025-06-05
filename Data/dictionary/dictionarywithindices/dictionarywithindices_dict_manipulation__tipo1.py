import json

def agregar_elemento(diccionario: dict, clave, valor) -> dict:
    diccionario[clave] = valor
    return diccionario

def actualizar_elemento(diccionario: dict, clave, valor) -> dict:
    if clave in diccionario:
        diccionario[clave] = valor
    return diccionario

def eliminar_elemento(diccionario: dict, clave) -> dict:
    diccionario.pop(clave, None)
    return diccionario

def verificar_clave(diccionario: dict, clave) -> bool:
    return clave in diccionario

def iterar_claves(diccionario: dict) -> list:
    return list(diccionario.keys())

def iterar_valores(diccionario: dict) -> list:
    return list(diccionario.values())

def combinar_diccionarios(dic1: dict, dic2: dict) -> dict:
    dic1.update(dic2)
    return dic1

def copiar_diccionario(diccionario: dict) -> dict:
    return diccionario.copy()

def limpiar_diccionario(diccionario: dict) -> dict:
    diccionario.clear()
    return diccionario

def encontrar_clave_por_valor(diccionario: dict, valor) -> list:
    return [k for k, v in diccionario.items() if v == valor]

def invertir_diccionario(diccionario: dict) -> dict:
    return {v: k for k, v in diccionario.items()}

def contar_valores(diccionario: dict) -> dict:
    conteos = {}
    for valor in diccionario.values():
        conteos[valor] = conteos.get(valor, 0) + 1
    return conteos

def filtrar_por_valor(diccionario: dict, condicion) -> dict:
    return {k: v for k, v in diccionario.items() if condicion(v)}

def claves_valor_min_max(diccionario: dict) -> tuple:
    if not diccionario:
        return None, None
    clave_min = min(diccionario, key=diccionario.get)
    clave_max = max(diccionario, key=diccionario.get)
    return clave_min, clave_max

def ordenar_por_valor(diccionario: dict, reverso=False) -> dict:
    return dict(sorted(diccionario.items(), key=lambda item: item[1], reverse=reverso))

def a_json(diccionario: dict) -> str:
    return json.dumps(diccionario)

def sumar_valores_numericos(diccionario: dict) -> float:
    return sum(v for v in diccionario.values() if isinstance(v, (int, float)))

if __name__ == "__main__":
    ejemplo_diccionario = {'nombre': 'Alice', 'edad': 30, 'salario': 50000, 'departamento': 'Ingeniería'}

    print("Diccionario Original:", ejemplo_diccionario)
    print("Después de Agregar Puesto:", agregar_elemento(ejemplo_diccionario.copy(), 'puesto', 'Ingeniera'))
    print("Después de Actualizar Edad:", actualizar_elemento(ejemplo_diccionario.copy(), 'edad', 31))
    print("Después de Eliminar Nombre:", eliminar_elemento(ejemplo_diccionario.copy(), 'nombre'))
    print("¿Existe la Edad?:", verificar_clave(ejemplo_diccionario, 'edad'))
    print("Iterando Claves:", iterar_claves(ejemplo_diccionario))
    print("Iterando Valores:", iterar_valores(ejemplo_diccionario))
    print("Combinando con Otro Diccionario:", combinar_diccionarios(ejemplo_diccionario.copy(), {'ciudad': 'Nueva York'}))
    print("Diccionario Copiado:", copiar_diccionario(ejemplo_diccionario))
    print("Diccionario Limpiado:", limpiar_diccionario(ejemplo_diccionario.copy()))

    print("Encontrar Clave por Valor (50000):", encontrar_clave_por_valor(ejemplo_diccionario, 50000))
    print("Diccionario Invertido:", invertir_diccionario(ejemplo_diccionario))
    print("Contar Valores:", contar_valores(ejemplo_diccionario))
    print("Filtrar por Valor (>10000):", filtrar_por_valor(ejemplo_diccionario, lambda x: isinstance(x, int) and x > 10000))
    print("Claves con Valor Mínimo y Máximo:", claves_valor_min_max({'a': 1, 'b': 2, 'c': 3}))
    print("Ordenado por Valor:", ordenar_por_valor({'a': 3, 'b': 1, 'c': 2}))
    print("Diccionario como JSON:", a_json(ejemplo_diccionario))
    print("Suma de Valores Numéricos:", sumar_valores_numericos(ejemplo_diccionario))