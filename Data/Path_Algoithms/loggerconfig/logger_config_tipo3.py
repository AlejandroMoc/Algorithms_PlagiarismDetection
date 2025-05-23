import logging

def configurar_logger():
    registrador = logging.getLogger('Algoritmos_Ruta')
    registrador.setLevel(logging.DEBUG)
    controlador = logging.StreamHandler()
    formateador = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    controlador.setFormatter(formateador)
    registrador.addHandler(controlador)
    return registrador

registrador = configurar_logger()