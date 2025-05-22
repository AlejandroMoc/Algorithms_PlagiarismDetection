python
import logging

def configurar_registro(nombre: str, nivel: int = logging.INFO) -> logging.Logger:
    registro = logging.getLogger(nombre)
    if not registro.handlers:
        manejador = logging.StreamHandler()
        formateador = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        manejador.setFormatter(formateador)
        registro.setLevel(nivel)
        registro.addHandler(manejador)
    return registro