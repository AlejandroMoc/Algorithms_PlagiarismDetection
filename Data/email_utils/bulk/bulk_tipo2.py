import time
from typing import List, Dict, Any
from .client import EmailClient
from .exceptions import TemplateError

class EmailDistributor:
    def __init__(
        self,
        transmitter: EmailClient,
        package_size: int = 50,
        pause: float = 1.0
    ):
        self.transmitter = transmitter
        self.package_size = package_size
        self.pause = pause

    def distribute_emails(
        self,
        receivers: List[Dict[str, Any]],
        theme: str,
        model: str
) -> List[Dict[str, Any]]:
        """
        Distribute emails using a model.
        
        Args:
            receivers: List of dictionaries containing receiver data
            theme: Email theme
            model: Email model with placeholders
            
        Returns:
            List of dictionaries with distribution results
        """
        results = []
        
        for i in range(0, len(receivers), self.package_size):
            batch = receivers[i:i + self.package_size]
            
            for receiver in batch:
                try:
                    # Format model with receiver data
                    email_content = model.format(**receiver)
                    
                    # Send email
                    self.transmitter.send_text_email(
                        to=receiver['email'],
                        subject=theme,
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
            
            # Pause between batches
            if i + self.package_size < len(receivers):
                time.sleep(self.pause)
        
        return results

    def distribute_html_emails(
        self,
        receivers: List[Dict[str, Any]],
        theme: str,
        html_model: str,
        text_model: str = None
    ) -> List[Dict[str, Any]]:
        """
        Distribute HTML emails using a model.
        
        Args:
            receivers: List of dictionaries containing receiver data
            theme: Email theme
            html_model: HTML email model
            text_model: Optional plain text model
        """
        results = []
        
        for i in range(0, len(receivers), self.package_size):
            batch = receivers[i:i + self.package_size]
            
            for receiver in batch:
                try:
                    # Format models with receiver data
                    html_content = html_model.format(**receiver)
                    text_content = text_model.format(**receiver) if text_model else None
                    
                    # Send email
                    self.transmitter.send_html_email(
                        to=receiver['email'],
                        subject=theme,
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
            
            # Pause between batches
            if i + self.package_size < len(receivers):
                time.sleep(self.pause)
        
        return results