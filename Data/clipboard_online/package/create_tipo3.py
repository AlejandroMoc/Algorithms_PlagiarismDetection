import asyncio
import requests
import subprocess
import platform


def obtener_contenido_portapapeles() -> str:
    sistema_operativo_actual = platform.system()

    try:
        if sistema_operativo_actual == "Linux":
            resultado = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"], stdout=subprocess.PIPE
            )
            return resultado.stdout.decode("utf-8")
        elif sistema_operativo_actual == "Darwin":
            resultado = subprocess.run(["pbpaste"], stdout=subprocess.PIPE)
            return resultado.stdout.decode("utf-8")
        elif sistema_operativo_actual == "Windows":
            resultado = subprocess.run(
                ["powershell", "-command", "Get-Clipboard"], stdout=subprocess.PIPE
            )
            return resultado.stdout.decode("utf-8")
        else:
            print(f"Sistema operativo no compatible: {sistema_operativo_actual}")
            return ""
    except subprocess.CalledProcessError:
        print("Error al obtener el contenido del portapapeles.")
        return ""


async def crear(url_id, tiempo_vida="60s"):
    texto = obtener_contenido_portapapeles()
    base_url = "https://cl1p.net"
    convertir_tiempo_a_segundos(tiempo_vida)

    async def crear_clip():
        url = f"{base_url}/{url_id}"
        datos = {"contenido": texto}
        try:
            respuesta = await asyncio.to_thread(requests.post, url, data=datos)
            if respuesta.status_code == 200:
                return True
            else:
                print(f"Fallo al crear el clip: Código de estado {respuesta.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Fallo al crear el clip: {e}")
            return False

    exito_creacion = await crear_clip()
    if exito_creacion:
        print(f"Clip creado exitosamente: https://cl1p.net/{url_id}")
    else:
        print("Fallo al crear el clip.")


def convertir_tiempo_a_segundos(tiempo_vida):
    partes = tiempo_vida.split()
    if len(partes) != 2:
        raise ValueError(
            "El tiempo de vida debe estar en el formato '<número> <unidad>' (por ejemplo, '1 día')."
        )

    valor, unidad = partes
    try:
        valor = int(valor)
    except ValueError:
        raise ValueError("La primera parte del tiempo de vida debe ser un entero.")

    if "segundo" in unidad or "segundos" in unidad:
        return valor
    elif "minuto" in unidad or "minutos" in unidad:
        return valor * 60
    elif "hora" in unidad or "horas" in unidad:
        return valor * 3600
    elif "día" in unidad or "días" in unidad:
        return valor * 86400
    elif "semana" in unidad or "semanas" in unidad:
        return valor * 604800
    else:
        raise ValueError(
            "Unidad de tiempo inválida. Utiliza segundos, minutos, horas, días o semanas."
        )


if __name__ == "__main__":
    asyncio.run(
        crear("yash1", "Este clip se autodestruirá en", "1 día")
    )