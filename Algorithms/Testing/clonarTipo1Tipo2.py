import os
import ast
import shutil
import random
import glob

# Número de clones por archivo original
CLONES_POR_ARCHIVO = 1

def agregar_comentarios(code):
    lineas = code.splitlines()
    nuevas = []
    for l in lineas:
        nuevas.append(l)
        if l.strip() and not l.strip().startswith("#") and random.random() < 0.2:
            nuevas.append("# Comentario añadido para clon")
    return "\n".join(nuevas)

class Renombrador(ast.NodeTransformer):
    def __init__(self):
        self.variables = {}
        self.contador = 0

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            if node.id not in self.variables:
                self.variables[node.id] = f"var_{self.contador}"
                self.contador += 1
            node.id = self.variables[node.id]
        elif isinstance(node.ctx, ast.Load) and node.id in self.variables:
            node.id = self.variables[node.id]
        return node

def renombrar_variables(code):
    try:
        tree = ast.parse(code)
        nuevo_tree = Renombrador().visit(tree)
        ast.fix_missing_locations(nuevo_tree)
        return ast.unparse(nuevo_tree)
    except Exception as e:
        print("Error al renombrar:", e)
        return code

def clonar_codigo(original_path, destino_path, tipo):
    with open(original_path, "r", encoding="utf-8") as f:
        code = f.read()

    for i in range(CLONES_POR_ARCHIVO):
        modificado = renombrar_variables(code)
        modificado = agregar_comentarios(modificado)
        base = os.path.basename(original_path).replace(".py", "")
        nuevo_nombre = f"{base}_clone{i+1}_tipo{tipo}.py"
        nueva_ruta = os.path.join(destino_path, nuevo_nombre)
        with open(nueva_ruta, "w", encoding="utf-8") as nf:
            nf.write(modificado)
        print(f"✅ Clon generado: {nueva_ruta}")

def procesar_carpeta(carpeta):
    archivos = glob.glob(os.path.join(carpeta, "*.py"))
    for archivo in archivos:
        archivo_lower = os.path.basename(archivo).lower()
        if "tipo1" in archivo_lower:
            clonar_codigo(archivo, carpeta, tipo=1)
        elif "tipo2" in archivo_lower:
            clonar_codigo(archivo, carpeta, tipo=2)

    # Si hay plagio_tipo1.txt, duplicarlo también para los clones
    flag_path = os.path.join(carpeta, "plagio_tipo1.txt")
    if os.path.exists(flag_path):
        with open(flag_path, "a") as f:
            f.write("# Clones generados automáticamente\n")

if __name__ == "__main__":
    BASE = "Data"  # Cambia si tu carpeta raíz se llama diferente
    for root, dirs, files in os.walk(BASE):
        if any(f.endswith(".py") and ("tipo1" in f.lower() or "tipo2" in f.lower()) for f in files):
            procesar_carpeta(root)

    print("\n🎉 Generación de clones completada.")
