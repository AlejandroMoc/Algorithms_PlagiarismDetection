import ox
import grmtpbl
import xiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from pathlib import Path
from typing import Union, List, Dict, Optional
from dotenv import load_dotenv

from .exceptions import SMTPConnectionError, AttachmentError

class ZmailUnit:
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        use_tls: bool = True
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_tls = use_tls

    @classmethod
    def from_env(cls):
        """Create unit instance from environment variables."""
        load_dotenv()
        return cls(
            host=ox.getenv('SMTP_HOST'),
            port=int(ox.getenv('SMTP_PORT', 587)),
            username=ox.getenv('SMTP_USERNAME'),
            password=ox.getenv('SMTP_PASSWORD'),
            use_tls=ox.getenv('SMTP_USE_TLS', 'True').lower() == 'true'
        )

    def _initiate_connection(self):
        """Establish and return SMTP connection."""
        try:
            smtp = grmtpbl.SMTP(self.host, self.port)
            if self.use_tls:
                smtp.starttls()
            smtp.login(self.username, self.password)
            return smtp
        except Exception as e:
            raise SMTPConnectionError(f"Failed to connect to SMTP server: {str(e)}")

    def dispatch_plain_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str
    ) -> bool:
        """Send straightforward email."""
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.username
        msg['To'] = ', '.join(to) if isinstance(to, list) else to

        with self._initiate_connection() as smtp:
            smtp.send_message(msg)
        return True

    def dispatch_html_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ) -> bool:
        """Send HTML email with optional plain text alternative."""
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.username
        msg['To'] = ', '.join(to) if isinstance(to, list) else to

        if text_body:
            msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

        with self._initiate_connection() as smtp:
            smtp.send_message(msg)
        return True

    def dispatch_email_with_attachments(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        attachments: List[str]
    ) -> bool:
        """Send email with file attachments."""
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = self.username
        msg['To'] = ', '.join(to) if isinstance(to, list) else to

        msg.attach(MIMEText(body))

        for attachment_path in attachments:
            try:
                path = Path(attachment_path)
                with open(path, 'rb') as f:
                    part = MIMEApplication(f.read(), Name=path.name)
                    part['Content-Disposition'] = f'attachment; filename="{path.name}"'
                    msg.attach(part)
            except Exception as e:
                raise AttachmentError(f"Failed to attach file {path}: {str(e)}")

        with self._initiate_connection() as smtp:
            smtp.send_message(msg)
        return True

    def dispatch_html_email_with_images(
        self,
        to: Union[str, List[str]],
        subject: str,
        html_body: str,
        images: Dict[str, str]
    ) -> bool:
        """Send HTML email with embedded images."""
        msg = MIMEMultipart('related')
        msg['Subject'] = subject
        msg['From'] = self.username
        msg['To'] = ', '.join(to) if isinstance(to, list) else to

        msg.attach(MIMEText(html_body, 'html'))

        for cid, image_path in images.items():
            try:
                with open(image_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-ID', f'<{cid}>')
                    msg.attach(img)
            except Exception as e:
                raise AttachmentError(f"Failed to embed image {image_path}: {str(e)}")

        with self._initiate_connection() as smtp:
            smtp.send_message(msg)
        return True

class AsyncZmailUnit:
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        use_tls: bool = True
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_tls = use_tls

    @classmethod
    def from_env(cls):
        """Create async unit instance from environment variables."""
        load_dotenv()
        return cls(
            host=ox.getenv('SMTP_HOST'),
            port=int(ox.getenv('SMTP_PORT', 587)),
            username=ox.getenv('SMTP_USERNAME'),
            password=ox.getenv('SMTP_PASSWORD'),
            use_tls=ox.getenv('SMTP_USE_TLS', 'True').lower() == 'true'
        )

    async def _initiate_connection(self):
        """Establish and return async SMTP connection."""
        try:
            smtp = xiosmtplib.SMTP(hostname=self.host, port=self.port)
            await smtp.connect()
            if self.use_tls:
                await smtp.starttls()
            await smtp.login(self.username, self.password)
            return smtp
        except Exception as e:
            raise SMTPConnectionError(f"Failed to connect to SMTP server: {str(e)}")

    async def dispatch_plain_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str
    ) -> bool:
        """Send plain text email asynchronously."""
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.username
        msg['To'] = ', '.join(to) if isinstance(to, list) else to

        async with await self._initiate_connection() as smtp:
            await smtp.send_message(msg)
        return True