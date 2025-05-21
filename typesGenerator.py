import os
import openai
from dotenv import load_dotenv

# Cargar la API Key desde .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_completion(prompt, model="gpt-3.5-turbo"):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Eres un experto en programación que transforma código manteniendo su funcionalidad."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content

def limpiar_formato_markdown(codigo):
    return "\n".join(
        linea for linea in codigo.splitlines() if not linea.strip().startswith("```")
    )

def generar_versiones_plagio(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        codigo = f.read()

    base, ext = os.path.splitext(ruta_archivo)

    # Tipo 1
    with open(f"{base}_original{ext}", "w", encoding="utf-8") as f:
        f.write(codigo)
    print(f"[✓] Tipo 1 (copia exacta): {base}_original{ext}")

    # Tipo 2
    prompt_tipo2 = (
        "Reescribe el siguiente código Python cambiando todos los nombres de funciones, variables y clases por otros completamente diferentes, "
        "usando nombres inventivos u originales (pueden ser palabras sin sentido, abreviaciones o combinaciones inusuales). "
        "Evita traducciones literales o nombres obvios como 'data', 'var' o 'counter'. "
        "No cambies la lógica ni el comportamiento del programa.\n\n"
        "Devuelve únicamente el código renombrado:\n\n"
        f"```python\n{codigo}\n```"
    )
    try:
        tipo2 = get_completion(prompt_tipo2)
        tipo2 = limpiar_formato_markdown(tipo2)
        with open(f"{base}_tipo2{ext}", "w", encoding="utf-8") as f:
            f.write(tipo2)
        print(f"[✓] Tipo 2 (renombramiento): {base}_tipo2{ext}")
    except Exception as e:
        print(f"[!] Error en tipo 2: {e}")

    # Tipo 3
    prompt_tipo3 = (
        "Reescribe el siguiente código Python manteniendo exactamente la misma funcionalidad, realizando cambios que modifiquen "
        "nombres de variables y funciones, pero conservando la estructura del programa.\n\n"
        "Transformaciones permitidas (estructurales y léxicas):\n"
        "- Cambiar nombres de variables y funciones por otros equivalentes.\n"
        "- Reordenar funciones en el archivo.\n"
        "- Reemplazar bucles for por while equivalentes (o viceversa) manteniendo la lógica.\n"
        "- Cambiar condicionales sin alterar la semántica (ej. invertir la lógica del if).\n"
        "- Separar instrucciones en múltiples pasos lógicos (ej. usar variables intermedias).\n"
        "- Añadir funciones auxiliares que encapsulen bloques existentes.\n"
        "- Reemplazar operaciones o estructuras de datos por otras equivalentes (ej. range(len(x)) por enumerate).\n\n"
        "Evita explicaciones, comentarios o texto adicional. Devuelve únicamente el código Python modificado:\n\n"
        f"python\n{codigo}\n"
    )
    try:
        tipo3 = get_completion(prompt_tipo3)
        tipo3 = limpiar_formato_markdown(tipo3)
        with open(f"{base}_tipo3{ext}", "w", encoding="utf-8") as f:
            f.write(tipo3)
        print(f"[✓] Tipo 3 (reestructuración): {base}_tipo3{ext}")
    except Exception as e:
        print(f"[!] Error en tipo 3: {e}")

def procesar_data():
    raiz = os.path.dirname(os.path.abspath(__file__))
    carpeta_data = os.path.join(raiz, "Data")
    for dirpath, _, filenames in os.walk(carpeta_data):
        for filename in filenames:
            if filename.endswith(".py") and not any(s in filename for s in ["_original", "_tipo2", "_tipo3"]):
                ruta = os.path.join(dirpath, filename)
                generar_versiones_plagio(ruta)

if __name__ == "__main__":
    procesar_data()
