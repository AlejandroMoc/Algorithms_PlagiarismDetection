python
import pickle
from typing import Any, Optional

def buscar_datos_en_archivo_binario(ruta_archivo: str, datos_busqueda: Any) -> Optional[Any]:
    if not isinstance(ruta_archivo, str):
        raise ValueError("La ruta del archivo debe ser un string.")
    
    try:
        with open(ruta_archivo, 'rb') as archivo:
            while True:
                try:
                    # Cargar cada objeto uno por uno
                    datos = pickle.load(archivo)
                    if datos == datos_busqueda:
                        return datos
                except EOFError:
                    # Fin del archivo alcanzado
                    break
        print("No se encontraron datos coincidentes.")
        return None
    except FileNotFoundError:
        print(f"Error: Archivo '{ruta_archivo}' no encontrado. Por favor verifica la ruta del archivo.")
    except pickle.PickleError as e:
        print(f"Error: Falló al deserializar los datos. Detalles: {e}")
    except Exception as e:
        print(f"Error: Ocurrió un error inesperado. Detalles: {e}")
        return None

if __name__ == "__main__":
    # Ejemplo de Uso
    ruta_archivo = r"pysnippets\Files\sample_data.pkl"
    datos_busqueda = {'nombre': 'Aryan Sharma',
        'edad': 24,
        'es_estudiante': True,
        'puntajes': [55, 88, 26],
        'detalles': {
            'curso': 'Ciencias de la Computación',
            'año': 2025
        }}
    
    resultado = buscar_datos_en_archivo_binario(ruta_archivo, datos_busqueda)
    if resultado is not None:
        print("Datos coincidentes encontrados:")
        print(resultado)
    else:
        print("No se encontraron datos coincidentes.")