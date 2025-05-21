import os
import difflib

def comparar_archivos(nombre_base, ext, dirpath, reporte):
    rutas = {
        "original": os.path.join(dirpath, f"{nombre_base}_original{ext}"),
        "tipo2": os.path.join(dirpath, f"{nombre_base}_tipo2{ext}"),
        "tipo3": os.path.join(dirpath, f"{nombre_base}_tipo3{ext}")
    }

    for tipo in ["tipo2", "tipo3"]:
        if not os.path.exists(rutas[tipo]):
            reporte.append(f"[✗] {tipo} no encontrado para {nombre_base}{ext} en {dirpath}\n")
            continue

        with open(rutas["original"], "r", encoding="utf-8") as f1, open(rutas[tipo], "r", encoding="utf-8") as f2:
            lineas1 = f1.readlines()
            lineas2 = f2.readlines()

        diff = list(difflib.unified_diff(
            lineas1,
            lineas2,
            fromfile=f"{nombre_base}_original{ext}",
            tofile=f"{nombre_base}_{tipo}{ext}",
            lineterm=''
        ))

        if diff:
            reporte.append(f"\n[≠] Diferencias entre {nombre_base}_original.py y {nombre_base}_{tipo}.py:\n")
            reporte.extend(diff)
            reporte.append("\n")
        else:
            reporte.append(f"[=] {nombre_base}_original.py y {nombre_base}_{tipo}.py son idénticos.\n")

def generar_reporte():
    raiz = os.path.dirname(os.path.abspath(__file__))
    carpeta_data = os.path.join(raiz, "Data")
    reporte = []

    for dirpath, _, filenames in os.walk(carpeta_data):
        for filename in filenames:
            if filename.endswith("_original.py"):
                nombre_base, ext = filename.rsplit("_original", 1)
                comparar_archivos(nombre_base, ext, dirpath, reporte)

    ruta_reporte = os.path.join(raiz, "reporte_diferencias.txt")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.writelines(linea if linea.endswith("\n") else linea + "\n" for linea in reporte)

    print(f"📄 Reporte generado: {ruta_reporte}")

if __name__ == "__main__":
    generar_reporte()
