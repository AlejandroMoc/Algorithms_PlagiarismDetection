import re

class ValidadorContrasena:
    def __init__(self, longitud_minima=8, longitud_maxima=64, requerir_mayuscula=True, requerir_minuscula=True, requerir_digitos=True, requerir_simbolos=True):
        self.longitud_minima = longitud_minima
        self.longitud_maxima = longitud_maxima
        self.requerir_mayuscula = requerir_mayuscula
        self.requerir_minuscula = requerir_minuscula
        self.requerir_digitos = requerir_digitos
        self.requerir_simbolos = requerir_simbolos

    def validar(self, contrasena):
        if not (self.longitud_minima <= len(contrasena) <= self.longitud_maxima):
            return False
        if self.requerir_mayuscula and not re.search(r'[A-Z]', contrasena):
            return False
        if self.requerir_minuscula and not re.search(r'[a-z]', contrasena):
            return False
        if self.requerir_digitos and not re.search(r'\d', contrasena):
            return False
        if self.requerir_simbolos and not re.search(r'[!@#$%^&*(),.?":{}|<>]', contrasena):
            return False
        return True