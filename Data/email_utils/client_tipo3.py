import os
import smtplib
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from pathlib import Path
from typing import Union, List, Dict, Optional
from dotenv import load_dotenv

from .exceptions import SMTPConnectionError, AttachmentError

class CorreoElectronico:
    def __init__(
        self,
        servidor: str,
        puerto: int,
        usuario: str,
        contraseña: str,
        usar_tls: bool = True
    ):
        self.servidor = servidor
        self.puerto = puerto
        self.usuario = usuario
        self.contraseña = contraseña
        self.usar_tls = usar_tls

    @classmethod
    def desde_entorno(cls):
        """Crear instancia de cliente desde variables de entorno."""
        load_dotenv()
        return cls(
            servidor=os.getenv('SMTP_HOST'),
            puerto=int(os.getenv('SMTP_PORT', 587)),
            usuario=os.getenv('SMTP_USERNAME'),
            contraseña=os.getenv('SMTP_PASSWORD'),
            usar_tls=os.getenv('SMTP_USE_TLS', 'True').lower() == 'true'
        )

    def _crear_conexion(self):
        """Crear y devolver conexión SMTP."""
        try:
            smtp = smtplib.SMTP(self.servidor, self.puerto)
            if self.usar_tls:
                smtp.starttls()
            smtp.login(self.usuario, self.contraseña)
            return smtp
        except Exception as e:
            raise SMTPConnectionError(f"No se pudo conectar al servidor SMTP: {str(e)}")

    def enviar_correo_texto(
        self,
        para: Union[str, List[str]],
        asunto: str,
        cuerpo: str
    ) -> bool:
        """Enviar correo de texto plano."""
        msg = MIMEText(cuerpo)
        msg['Subject'] = asunto
        msg['From'] = self.usuario
        msg['To'] = ', '.join(para) if isinstance(para, list) else para

        with self._crear_conexion() as smtp:
            smtp.send_message(msg)
        return True

    def enviar_correo_html(
        self,
        para: Union[str, List[str]],
        asunto: str,
        cuerpo_html: str,
        cuerpo_texto: Optional[str] = None
    ) -> bool:
        """Enviar correo HTML con texto alternativo opcional."""
        msg = MIMEMultipart('alternative')
        msg['Subject'] = asunto
        msg['From'] = self.usuario
        msg['To'] = ', '.join(para) if isinstance(para, list) else para

        if cuerpo_texto:
            msg.attach(MIMEText(cuerpo_texto, 'plain'))
        msg.attach(MIMEText(cuerpo_html, 'html'))

        with self._crear_conexion() as smtp:
            smtp.send_message(msg)
        return True

    def enviar_correo_con_adjuntos(
        self,
        para: Union[str, List[str]],
        asunto: str,
        cuerpo: str,
        adjuntos: List[str]
    ) -> bool:
        """Enviar correo con adjuntos."""
        msg = MIMEMultipart()
        msg['Subject'] = asunto
        msg['From'] = self.usuario
        msg['To'] = ', '.join(para) if isinstance(para, list) else para

        msg.attach(MIMEText(cuerpo))

        for ruta_adjunto in adjuntos:
            try:
                ruta = Path(ruta_adjunto)
                with open(ruta, 'rb') as f:
                    parte = MIMEApplication(f.read(), Name=ruta.name)
                    parte['Content-Disposition'] = f'attachment; filename="{ruta.name}"'
                    msg.attach(parte)
            except Exception as e:
                raise AttachmentError(f"No se pudo adjuntar el archivo {ruta}: {str(e)}")

        with self._crear_conexion() as smtp:
            smtp.send_message(msg)
        return True

    def enviar_correo_html_con_imagenes(
        self,
        para: Union[str, List[str]],
        asunto: str,
        cuerpo_html: str,
        imagenes: Dict[str, str]
    ) -> bool:
        """Enviar correo HTML con imágenes incrustadas."""
        msg = MIMEMultipart('related')
        msg['Subject'] = asunto
        msg['From'] = self.usuario
        msg['To'] = ', '.join(para) if isinstance(para, list) else para

        msg.attach(MIMEText(cuerpo_html, 'html'))

        for cid, ruta_imagen in imagenes.items():
            try:
                with open(ruta_imagen, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-ID', f'<{cid}>')
                    msg.attach(img)
            except Exception as e:
                raise AttachmentError(f"No se pudo incrustar la imagen {ruta_imagen}: {str(e)}")

        with self._crear_conexion() as smtp:
            smtp.send_message(msg)
        return True

class ClienteCorreoElectronicoAsincronico:
    def __init__(
        self,
        servidor: str,
        puerto: int,
        usuario: str,
        contraseña: str,
        usar_tls: bool = True
    ):
        self.servidor = servidor
        self.puerto = puerto
        self.usuario = usuario
        self.contraseña = contraseña
        self.usar_tls = usar_tls

    @classmethod
    def desde_entorno(cls):
        """Crear instancia de cliente asíncrono desde variables de entorno."""
        load_dotenv()
        return cls(
            servidor=os.getenv('SMTP_HOST'),
            puerto=int(os.getenv('SMTP_PORT', 587)),
            usuario=os.getenv('SMTP_USERNAME'),
            contraseña=os.getenv('SMTP_PASSWORD'),
            usar_tls=os.getenv('SMTP_USE_TLS', 'True').lower() == 'true'
        )

    async def _crear_conexion(self):
        """Crear y devolver conexión SMTP asíncrona."""
        try:
            smtp = aiosmtplib.SMTP(hostname=self.servidor, port=self.puerto)
            await smtp.connect()
            if self.usar_tls:
                await smtp.starttls()
            await smtp.login(self.usuario, self.contraseña)
            return smtp
        except Exception as e:
            raise SMTPConnectionError(f"No se pudo conectar al servidor SMTP: {str(e)}")

    async def enviar_correo_texto(
        self,
        para: Union[str, List[str]],
        asunto: str,
        cuerpo: str
    ) -> bool:
        """Enviar correo de texto plano de forma asíncrona."""
        msg = MIMEText(cuerpo)
        msg['Subject'] = asunto
        msg['From'] = self.usuario
        msg['To'] = ', '.join(para) if isinstance(para, list) else para

        async with await self._crear_conexion() as smtp:
            await smtp.send_message(msg)
        return True