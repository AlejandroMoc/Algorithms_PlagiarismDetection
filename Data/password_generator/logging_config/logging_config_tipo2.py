import logging

registra = logging.getLogger('generador_claves')
registra.setLevel(logging.INFO)
manejador = logging.StreamHandler()
formateador = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
manejador.setFormatter(formateador)
registra.addHandler(manejador)

def configurar_registro():
    return registra