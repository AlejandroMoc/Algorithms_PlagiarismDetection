import pickle
from typing import Any

def guardar_archivo_binario(ruta_archivo: str, datos: Any) -> None:
    if not isinstance(ruta_archivo, str):
        raise ValueError("La ruta del archivo debe ser un string.")
    try:
        with open(ruta_archivo, 'wb') as archivo:
            pickle.dump(datos, archivo)
        print(f"Datos guardados exitosamente en '{ruta_archivo}'")
    except pickle.PickleError as e:
        print(f"Error: Fallo al serializar los datos. Detalles: {e}")
    except Exception as e:
        print(f"Error: Ocurrió un error inesperado. Detalles: {e}")

if __name__ == "__main__":
    # Ejemplo de Uso
    ruta_archivo = r"pysnippets\Files\datos_ejemplo.pkl"
    datos_a_guardar = {
        'nombre': 'Juan Pérez',
        'edad': 30,
        'es_estudiante': False,
        'calificaciones': [75, 90, 82],
        'detalles': {
            'carrera': 'Matemáticas',
            'año': 2023
        }
    }
    
    guardar_archivo_binario(ruta_archivo, datos_a_guardar)
    print("Operación de escritura completada.")