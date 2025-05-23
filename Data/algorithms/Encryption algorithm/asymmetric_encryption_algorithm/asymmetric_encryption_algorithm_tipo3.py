from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

# Generar par de claves RSA
def generar_par_claves():
    clave_privada = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    clave_publica = clave_privada.public_key()
    return clave_privada, clave_publica

# Encriptar datos usando clave pública
def encriptar(clave_publica, mensaje):
    texto_cifrado = clave_publica.encrypt(
        mensaje,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return texto_cifrado

# Desencriptar datos usando clave privada
def desencriptar(clave_privada, texto_cifrado):
    texto_plano = clave_privada.decrypt(
        texto_cifrado,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return texto_plano

# Serializar la clave privada para guardarla en un archivo
def serializar_clave_privada(clave_privada, contraseña=None):
    pem = clave_privada.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.BestAvailableEncryption(contraseña) if contraseña else serialization.NoEncryption()
    )
    return pem

# Serializar la clave pública para guardarla en un archivo
def serializar_clave_publica(clave_publica):
    pem = clave_publica.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return pem

# Ejemplo de uso
if __name__ == "__main__":
    mensaje = b"Ejemplo de encriptacion asimetrica con RSA"
    
    # Generar par de claves RSA
    clave_privada, clave_publica = generar_par_claves()

    # Encriptar el mensaje
    texto_cifrado = encriptar(clave_publica, mensaje)
    print("Cifrado:", texto_cifrado)
    
    # Desencriptar el mensaje
    mensaje_desencriptado = desencriptar(clave_privada, texto_cifrado)
    print("Desencriptado:", mensaje_desencriptado.decode('utf-8'))

    # Opcional: Serializar claves si se desea guardarlas
    clave_privada_pem = serializar_clave_privada(clave_privada, contraseña=b'micontrasena')
    clave_publica_pem = serializar_clave_publica(clave_publica)

    print("\nClave Privada formato PEM:\n", clave_privada_pem.decode('utf-8'))
    print("\nClave Pública formato PEM:\n", clave_publica_pem.decode('utf-8'))