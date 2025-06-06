import os

# Ruta base donde están los archivos tipo0
tipo0_base = r"C:\Users\Jacqui\Desktop\FinalAplicaciones\Algorithms_PlagiarismDetection\Data"
tipo1_generados = 0

for root, dirs, files in os.walk(tipo0_base):
    tipo0_files = [f for f in files if f.endswith(".py") and "tipo0" in f.lower()]
    
    for i, f in enumerate(tipo0_files):
        source_path = os.path.join(root, f)
        base_name = os.path.splitext(f)[0].replace("tipo0", "")
        new_name = f"{base_name}_tipo1.py"
        target_path = os.path.join(root, new_name)
        
        with open(source_path, 'r', encoding='utf-8') as src, open(target_path, 'w', encoding='utf-8') as dst:
            content = src.read()
            dst.write(content)
            tipo1_generados += 1

print(f"{tipo1_generados} archivos tipo 1 generados.")
