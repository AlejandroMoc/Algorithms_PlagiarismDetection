import os

def borrar_archivos_no_py(carpeta_raiz):
    for carpeta_actual, subcarpetas, lista_archivos in os.walk(carpeta_raiz):
        for archivo in lista_archivos:
            ruta_completa = os.path.join(carpeta_actual, archivo)
            if not archivo.endswith('.py'):
                try:
                    os.remove(ruta_completa)
                    print(f"Eliminado: {ruta_completa}")
                except Exception as e:
                    print(f"Error al eliminar {ruta_completa}: {e}")

if __name__ == "__main__":
    carpeta_origen = os.path.dirname(os.path.abspath(__file__))  # carpeta donde está este script
    borrar_archivos_no_py(carpeta_origen)