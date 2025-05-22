import os

def eliminar_no_py(directorio_base):
    for carpeta_actual, subcarpetas, archivos in os.walk(directorio_base):
        for archivo in archivos:
            ruta_completa = os.path.join(carpeta_actual, archivo)
            if not archivo.endswith('.py'):
                try:
                    os.remove(ruta_completa)
                    print(f"Eliminado: {ruta_completa}")
                except Exception as e:
                    print(f"Error al eliminar {ruta_completa}: {e}")

if __name__ == "__main__":
    carpeta_origen = os.path.dirname(os.path.abspath(__file__))  # carpeta donde está este script
    eliminar_no_py(carpeta_origen)
