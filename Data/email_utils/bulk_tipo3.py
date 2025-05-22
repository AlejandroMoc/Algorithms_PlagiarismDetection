import time
from typing import List, Dict, Any
from .client import EmailClient
from .exceptions import TemplateError

class SenderBulkEmails:
    def __init__(
        self,
        client: EmailClient,
        size_batch: int = 50,
        delay_time: float = 1.0
    ):
        self.client = client
        self.size_batch = size_batch
        self.delay_time = delay_time

    def send_emails_bulk(
        self,
        receivers: List[Dict[str, Any]],
        sub: str,
        template_email: str
) -> List[Dict[str, Any]]:
        results = []
        
        for i in range(0, len(receivers), self.size_batch):
            batch = receivers[i:i + self.size_batch]
            
            for receiver in batch:
                try:
                    email_content = template_email.format(**receiver)
                    
                    self.client.send_text_email(
                        to=receiver['email'],
                        subject=sub,
                        body=email_content
                    )
                    
                    results.append({
                        'email': receiver['email'],
                        'status': 'success',
                        'error': None
                    })
                except Exception as e:
                    results.append({
                        'email': receiver['email'],
                        'status': 'failed',
                        'error': str(e)
                    })
            
            if i + self.size_batch < len(receivers):
                time.sleep(self.delay_time)
        
        return results

    def send_emails_bulk_html(
        self,
        receivers: List[Dict[str, Any]],
        sub: str,
        html_template_email: str,
        text_template_email: str = None
    ) -> List[Dict[str, Any]]:
        results = []
        
        for i in range(0, len(receivers), self.size_batch):
            batch = receivers[i:i + self.size_batch]
            
            for receiver in batch:
                try:
                    html_content = html_template_email.format(**receiver)
                    text_content = text_template_email.format(**receiver) if text_template_email else None
                    
                    self.client.send_html_email(
                        to=receiver['email'],
                        subject=sub,
                        html_body=html_content,
                        text_body=text_content
                    )
                    
                    results.append({
                        'email': receiver['email'],
                        'status': 'success',
                        'error': None
                    })
                except Exception as e:
                    results.append({
                        'email': receiver['email'],
                        'status': 'failed',
                        'error': str(e)
                    })
            
            if i + self.size_batch < len(receivers):
                time.sleep(self.delay_time)
        
        return results