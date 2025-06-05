import shutil
import os

# Carpeta original de la base de datos
original_base_path = "Data"
output_base_path = "Data_balanceada"

# Palabras que generan ambigüedad
ambiguous_keywords = ["copiatipo1", "tipo0tipo1", "mezclado", "copiatipo"]

# Crear nueva base balanceada y limpia
if not os.path.exists(output_base_path):
    shutil.copytree(original_base_path, output_base_path)

# Duplicar archivos tipo 1 y corregir ambigüedades
duplicados = 0
corregidos = 0
for root, dirs, files in os.walk(output_base_path):
    for file in files:
        filepath = os.path.join(root, file)
        filename = file.lower()

        # Corregir nombres ambiguos que contienen múltiples tipos
        if any(k in filename for k in ambiguous_keywords):
            nuevo_nombre = file
            for palabra in ambiguous_keywords:
                nuevo_nombre = nuevo_nombre.replace(palabra, "")
            if nuevo_nombre != file:
                new_path = os.path.join(root, nuevo_nombre)
                os.rename(filepath, new_path)
                filepath = new_path
                corregidos += 1

        # Duplicar archivos tipo 1 si el nombre lo contiene explícitamente
        if "tipo1" in filename and filepath.endswith(".py"):
            for i in range(2):  # Duplicar 2 veces
                nuevo_nombre = f"copia_extra_tipo1_{i}_{file}"
                nuevo_path = os.path.join(root, nuevo_nombre)
                shutil.copy(filepath, nuevo_path)
                duplicados += 1

(corregidos, duplicados)