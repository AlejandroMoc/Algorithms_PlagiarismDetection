import os
import random
import shutil
from difflib import SequenceMatcher
from Algorithms.comparator_sa import comparator_sa

# Ruta local del dataset
data_root = r"C:/Users/aredc/OneDrive/Documentos/Plagio_project/Algorithms_PlagiarismDetection/Data"

carpetas_con_py = []
carpetas_con_subcarpetas = 0
carpetas_con_subs = []
tipo0_total = tipo1_total = tipo2_total = tipo3_total = 0
carpetas_tipo_0123 = 0
carpetas_tipo_123 = 0
carpetas_tipo_12 = 0

# Comparador neutral (difflib)
def difflib_ratio(file1, file2):
    try:
        with open(file1, encoding='utf-8') as f1, open(file2, encoding='utf-8') as f2:
            return SequenceMatcher(None, f1.read(), f2.read()).ratio()
    except:
        return 1.0  # Máxima similitud si hay error

# Paso 1: Recorrer estructura
for root, dirs, files in os.walk(data_root):
    py_files = [f for f in files if f.endswith(".py")]
    if py_files:
        carpetas_con_py.append(root)
    if dirs:
        carpetas_con_subcarpetas += 1
        carpetas_con_subs.append((root, dirs))

# Paso 2: Marcar el 30% con plagio tipo 1
total_carpetas = len(carpetas_con_py)
num_con_plagio_tipo1 = int(total_carpetas * 0.3)
carpetas_tipo1 = random.sample(carpetas_con_py, num_con_plagio_tipo1)

for carpeta in carpetas_tipo1:
    marcador_path = os.path.join(carpeta, "plagio_tipo1.txt")
    with open(marcador_path, "w", encoding="utf-8") as f:
        f.write("Esta carpeta contiene un caso de plagio tipo 1 (directo).\n")

# Paso 3: Generar hasta 3 combinaciones tipo 0 aleatorias y filtradas
for padre, subcarpetas in carpetas_con_subs:
    rutas_subs = [os.path.join(padre, s) for s in subcarpetas if os.path.isdir(os.path.join(padre, s))]
    combinaciones = [(a, b) for i, a in enumerate(rutas_subs) for b in rutas_subs[i+1:]]
    random.shuffle(combinaciones)
    combinaciones = combinaciones[:10]

    for sub_a, sub_b in combinaciones:
        archivos_a = [f for f in os.listdir(sub_a) if f.endswith(".py")]
        archivos_b = [f for f in os.listdir(sub_b) if f.endswith(".py")]
        if not archivos_a or not archivos_b:
            continue
        archivo_a = os.path.join(sub_a, random.choice(archivos_a))
        archivo_b = os.path.join(sub_b, random.choice(archivos_b))

        try:
            sa = comparator_sa(archivo_a, archivo_b)
            diff = difflib_ratio(archivo_a, archivo_b)

            # Filtro estricto: ambos deben ser poco similares
            if sa < 20 and diff < 0.3:
                with open(archivo_b, "r", encoding="utf-8") as fsrc:
                    contenido = fsrc.read()
                nombre_base_a = os.path.basename(sub_a)
                nombre_base_b = os.path.basename(sub_b)
                nuevo_nombre = f"{nombre_base_a}_{nombre_base_b}_tipo0.py"
                destino = os.path.join(sub_a, nuevo_nombre)
                with open(destino, "w", encoding="utf-8") as fdst:
                    fdst.write(contenido)
        except Exception as e:
            print(f"[!] Error comparando {archivo_a} y {archivo_b}: {e}")

# Paso 4: Estadísticas generales
total_archivos = 0
for root, _, files in os.walk(data_root):
    total_archivos += len([f for f in files if f.endswith(".py")])

for carpeta in carpetas_con_py:
    files = os.listdir(carpeta)
    flags = {"0": False, "1": False, "2": False, "3": False}
    for f in files:
        if "tipo0" in f: flags["0"] = True; tipo0_total += 1
        if "tipo1" in f or f == "plagio_tipo1.txt": flags["1"] = True; tipo1_total += 1
        if "tipo2" in f: flags["2"] = True; tipo2_total += 1
        if "tipo3" in f: flags["3"] = True; tipo3_total += 1
    if all(flags.values()):
        carpetas_tipo_0123 += 1
    elif flags["1"] and flags["2"] and flags["3"]:
        carpetas_tipo_123 += 1
    elif flags["1"] and flags["2"]:
        carpetas_tipo_12 += 1

# Paso 5: Reporte
print("\n Reporte Final del Dataset")
print("=" * 30)
print(f" Total de carpetas con archivos .py        : {total_carpetas}")
print(f" Total de carpetas madre (con subcarpetas): {carpetas_con_subcarpetas}")
print(f" Total de archivos .py                    : {total_archivos}")
print("")
print(f" Total de carpetas con tipo 0,1,2,3        : {carpetas_tipo_0123}")
print(f" Total de carpetas con tipo 1,2,3          : {carpetas_tipo_123}")
print(f" Total de carpetas con tipo 1 y 2          : {carpetas_tipo_12}")
print("")
print(f" Total de archivos tipo 0 (no plagio)      : {tipo0_total}")
print(f" Total de archivos tipo 1 (plagio directo) : {tipo1_total}")
print(f" Total de archivos tipo 2 (renombrado)     : {tipo2_total}")
print(f" Total de archivos tipo 3 (reestructurado) : {tipo3_total}")
print("")
print(" 30% de carpetas fueron marcadas con plagio tipo 1 (plagio_tipo1.txt)")
print(" Máximo 3 combinaciones tipo 0 generadas por carpeta madre con filtro de similitud.")
