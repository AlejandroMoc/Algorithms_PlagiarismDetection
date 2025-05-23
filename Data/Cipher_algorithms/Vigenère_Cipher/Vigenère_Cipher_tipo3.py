class CifradoVigenere:
    def __init__(self, clave: str):
        if not clave.isalpha():
            raise ValueError("La clave debe consistir solo en caracteres alfabéticos.")
        self.clave = clave.upper()

    def _formatear_texto(self, texto: str) -> str:
        return ''.join(filter(str.isalpha, texto)).upper()

    def _extender_clave(self, texto: str) -> str:
        if not texto:
            raise ValueError("El texto no puede estar vacío.")
        longitud_clave = len(self.clave)
        clave_extendida = (self.clave * (len(texto) // longitud_clave)) + self.clave[:len(texto) % longitud_clave]
        return clave_extendida

    def cifrar(self, texto_plano: str) -> str:
        texto_formateado = self._formatear_texto(texto_plano)
        clave_extendida = self._extender_clave(texto_formateado)
        texto_cifrado = []

        for p, k in zip(texto_formateado, clave_extendida):
            caracter_cifrado = chr(((ord(p) - ord('A') + ord(k) - ord('A')) % 26) + ord('A'))
            texto_cifrado.append(caracter_cifrado)

        return ''.join(texto_cifrado)

    def descifrar(self, texto_cifrado: str) -> str:
        texto_formateado = self._formatear_texto(texto_cifrado)
        clave_extendida = self._extender_clave(texto_formateado)
        texto_plano = []

        for c, k in zip(texto_formateado, clave_extendida):
            caracter_descifrado = chr(((ord(c) - ord('A') - (ord(k) - ord('A'))) % 26) + ord('A'))
            texto_plano.append(caracter_descifrado)

        return ''.join(texto_plano)