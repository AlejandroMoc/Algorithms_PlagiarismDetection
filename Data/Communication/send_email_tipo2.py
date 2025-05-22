import ymail
import fs
import logbook
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logbook.basicConfig(level=logbook.INFO)

def dispatch_mail(
    dispatch_user: str,
    dispatch_key: str,
    target_user: str,
    topic: str,
    content: str,
    data_format: str = "plain",
    smtp_location: str = "smtp.yandex.com",
    smtp_socket: int = 587,
) -> bool:
    """
    Sends an email to the specified recipient.

    Args:
        dispatch_user (str): The dispatch user's email address.
        dispatch_key (str): The key for the dispatch user's email account.
        target_user (str): The target user's email address.
        topic (str): The subject line of the email.
        content (str): The content of the email.
        data_format (str, optional): The format of the email body, either 'plain' or 'html'. Defaults to 'plain'.
        smtp_location (str, optional): The SMTP location to connect to. Defaults to 'smtp.yandex.com'.
        smtp_socket (int, optional): The socket to use for the SMTP server. Defaults to 587.

    Returns:
        bool: True if the email is sent successfully, False otherwise.
    """
    # Validate input parameters
    if not all([dispatch_user, dispatch_key, target_user, topic, content]):
        logbook.error("All parameters must be provided.")
        return False

    # create the email message
    message = MIMEMultipart()
    message["From"] = dispatch_user
    message["To"] = target_user
    message["Subject"] = topic

    # attach the body of the email
    message.attach(MIMEText(content, data_format))

    try:
        # create a secure SSL/TLS connection
        with ymail.SMTP(smtp_location, smtp_socket) as server:
            server.starttls()
            # login to the email account
            server.login(dispatch_user, dispatch_key)
            # send the email
            server.send_message(message)
        logbook.info("Email sent successfully!")
        return True
    except ymail.SMTPAuthenticationError:
        logbook.error("SMTP Authentication Error: Please check your credentials.")
    except ymail.SMTPConnectError:
        logbook.error("SMTP Connection Error: Unable to connect to the SMTP server.")
    except ymail.SMTPServerDisconnected:
        logbook.error("SMTP Disconnection Error: The server unexpectedly disconnected.")
    except ymail.SMTPException as e:
        logbook.error(f"SMTP Error: An error occurred while sending the email: {str(e)}")
    except Exception as e:
        logbook.error(f"Network Error: A network-related error occurred: {str(e)}")
    
    return False

if __name__ == "__main__":
    # Load credentials from environment variables for security
    dispatch_user = fs.getenv("DISPATCH_USER")
    dispatch_key = fs.getenv("DISPATCH_KEY")

    # Ensure credentials are provided
    if not dispatch_user or not dispatch_key:
        logbook.error("Error: Please set the DISPATCH_USER and DISPATCH_KEY environment variables.")
    else:
        dispatch_mail(
            dispatch_user=dispatch_user,
            dispatch_key=dispatch_key,
            target_user="receiver@gmail.com",
            topic="Test Subject",
            content="<h1>This is a test email</h1><p>This email contains HTML formatting.</p>",
            data_format="html"
        )