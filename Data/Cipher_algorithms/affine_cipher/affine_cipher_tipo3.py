class CifradoAfín:
    """
    Una clase para implementar el Cifrado Afín para encriptación y desencriptación.
    """
    
    LONGITUD_ALFABETO = 26  # Longitud del alfabeto

    def __init__(self, a, b):
        """
        Inicializa el Cifrado Afín con coeficientes especificados.

        Parámetros:
        - a (int): La clave multiplicativa (debe ser coprimo con LONGITUD_ALFABETO).
        - b (int): La clave aditiva.
        """
        self.m = self.LONGITUD_ALFABETO  # Inicializa m aquí
        if not self.es_clave_valida(a):
            raise ValueError(f"El valor de 'a' ({a}) debe ser coprimo con {self.m}.")
        
        self.a = a
        self.b = b

    def es_clave_valida(self, a):
        """
        Comprueba si 'a' es coprimo con la longitud del alfabeto.

        Parámetros:
        - a (int): La clave multiplicativa.

        Devuelve:
        - bool: True si 'a' es coprimo con LONGITUD_ALFABETO, False en caso contrario.
        """
        return self.mcd(a, self.m) == 1

    def mcd(self, x, y):
        """Calcula el máximo común divisor de x e y."""
        while y:
            x, y = y, x % y
        return x

    def encriptar(self, texto_plano):
        """
        Encripta el texto plano usando el cifrado afín.

        Parámetros:
        - texto_plano (str): El texto a encriptar.

        Devuelve:
        - str: El texto cifrado encriptado.
        """
        return ''.join(
            chr(((self.a * (ord(caracter) - ord('A')) + self.b) % self.m) + ord('A'))
            if caracter.isalpha() else caracter for caracter in texto_plano.upper()
        )

    def desencriptar(self, texto_cifrado):
        """
        Desencripta el texto cifrado usando el cifrado afín.

        Parámetros:
        - texto_cifrado (str): El texto a desencriptar.

        Devuelve:
        - str: El texto plano desencriptado.
        """
        a_inv = pow(self.a, -1, self.m)  # Inverso multiplicativo modular de a
        return ''.join(
            chr(((a_inv * ((ord(caracter) - ord('A')) - self.b)) % self.m) + ord('A'))
            if caracter.isalpha() else caracter for caracter in texto_cifrado.upper()
        )