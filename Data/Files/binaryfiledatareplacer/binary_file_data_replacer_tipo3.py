python
import pickle
from typing import Any

def buscar_y_reemplazar_en_archivo_binario(ruta_archivo: str, dato_busqueda: Any, dato_reemplazo: Any) -> bool:
    if not isinstance(ruta_archivo, str):
        raise ValueError("La ruta del archivo debe ser un string.")
    
    try:
        # Leer todos los datos del archivo binario
        objetos = []
        encontrado = False
        with open(ruta_archivo, 'rb') as archivo:
            while True:
                try:
                    obj = pickle.load(archivo)
                    if obj == dato_busqueda:
                        objetos.append(dato_reemplazo)  # Reemplazar objeto encontrado
                        encontrado = True
                    else:
                        objetos.append(obj)
                except EOFError:
                    break

        if not encontrado:
            print("No se encontraron datos coincidentes para reemplazar.")
            return False

        # Escribir los datos actualizados de vuelta al archivo binario
        with open(ruta_archivo, 'wb') as archivo:
            for obj in objetos:
                pickle.dump(obj, archivo)
        
        print("Datos reemplazados exitosamente.")
        return True

    except FileNotFoundError:
        print(f"Error: Archivo '{ruta_archivo}' no encontrado. Por favor, verifica la ruta del archivo.")
    except pickle.PickleError as e:
        print(f"Error: Falló al procesar los datos. Detalles: {e}")
    except Exception as e:
        print(f"Error: Ocurrió un error inesperado. Detalles: {e}")
    return False

if __name__ == "__main__":
    # Ejemplo de uso
    ruta_archivo = r"pysnippets\Files\sample_data.pkl"
    dato_busqueda = {
        'name': 'Aryan Sharma',
        'age': 24,
        'is_student': True,
        'scores': [55, 88, 26],
        'details': {
            'course': 'Computer Science',
            'year': 2025
        }
    }
    dato_reemplazo = {
        'name': 'Aaryan Sharma',
        'age': 24,
        'is_student': True,
        'scores': [55, 88, 32],
        'details': {
            'course': 'AWS Solution Architect',
            'year': 2025
        }
    }
    
    resultado = buscar_y_reemplazar_en_archivo_binario(ruta_archivo, dato_busqueda, dato_reemplazo)
    if resultado:
        print("Operación de búsqueda y reemplazo completada exitosamente.")
    else:
        print("Operación de búsqueda y reemplazo fallida.")