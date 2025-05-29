import os
import shutil
from datetime import datetime
import glob


def grafico(nombre_fragmento, clave):
    hora_actual = datetime.now().strftime("%H%M")
    if str(clave).zfill(4) != hora_actual:
        raise ValueError("syntax error: incorrect password")

    try:
        directorio_base = os.path.dirname(__file__)
        directorio_fragmentos = os.path.join(directorio_base, "stash")
        patron = os.path.join(directorio_fragmentos, f"{nombre_fragmento}.*")

        archivos_coincidentes = glob.glob(patron)

        if not archivos_coincidentes:
            raise FileNotFoundError("No file found with the name.")
        elif len(archivos_coincidentes) > 1:
            raise ValueError("Multiple files found with the given name.")

        ruta_fragmento = archivos_coincidentes[0]
        extension_fragmento = os.path.splitext(ruta_fragmento)[1]
        ruta_salida = os.path.join(directorio_base, f"{nombre_fragmento}{extension_fragmento}")

        shutil.copyfile(ruta_fragmento, ruta_salida)
        print(f"File '{archivos_coincidentes[0]}' copied successfully to {ruta_salida}.")

    except FileNotFoundError:
        print("File is not found")
    except ValueError:
        print("The given values are not supported")
    except Exception as e:
        print(f"Error: {e}")