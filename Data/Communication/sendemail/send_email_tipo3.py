import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logging.basicConfig(level=logging.INFO)

def enviar_correo(
    correo_emisor: str,
    contraseña_emisor: str,
    correo_destinatario: str,
    asunto: str,
    cuerpo: str,
    tipo_contenido: str = "plain",
    servidor_smtp: str = "smtp.gmail.com",
    puerto_smtp: int = 587,
) -> bool:
    if not all([correo_emisor, contraseña_emisor, correo_destinatario, asunto, cuerpo]):
        logging.error("Todos los parámetros deben ser proporcionados.")
        return False

    mensaje = MIMEMultipart()
    mensaje["From"] = correo_emisor
    mensaje["To"] = correo_destinatario
    mensaje["Subject"] = asunto
    mensaje.attach(MIMEText(cuerpo, tipo_contenido))

    try:
        with smtplib.SMTP(servidor_smtp, puerto_smtp) as servidor:
            servidor.starttls()
            servidor.login(correo_emisor, contraseña_emisor)
            servidor.send_message(mensaje)
        logging.info("¡Correo enviado con éxito!")
        return True
    except smtplib.SMTPAuthenticationError:
        logging.error("Error de autenticación SMTP: Por favor, verifique sus credenciales.")
    except smtplib.SMTPConnectError:
        logging.error("Error de conexión SMTP: No se puede conectar al servidor SMTP.")
    except smtplib.SMTPServerDisconnected:
        logging.error("Error de desconexión SMTP: El servidor se desconectó inesperadamente.")
    except smtplib.SMTPException as e:
        logging.error(f"Error SMTP: Se produjo un error al enviar el correo: {str(e)}")
    except Exception as e:
        logging.error(f"Error de red: Se produjo un error relacionado con la red: {str(e)}")
    
    return False

if __name__ == "__main__":
    correo_emisor = os.getenv("SENDER_EMAIL")
    contraseña_emisor = os.getenv("SENDER_PASSWORD")

    if not correo_emisor or not contraseña_emisor:
        logging.error("Error: Por favor, establezca las variables de entorno SENDER_EMAIL y SENDER_PASSWORD.")
    else:
        enviar_correo(
            correo_emisor=correo_emisor,
            contraseña_emisor=contraseña_emisor,
            correo_destinatario="receiver@gmail.com",
            asunto="Asunto de prueba",
            cuerpo="<h1>Este es un correo de prueba</h1><p>Este correo contiene formato HTML.</p>",
            tipo_contenido="html"
        )