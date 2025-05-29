import asyncio
import requests
import subprocess
import platform


def obtain_secret_data() -> str:
    # Detect the operating system
    current_system = platform.system()

    try:
        if current_system == "Linux":
            # Use xclip to obtain secret data
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"], stdout=subprocess.PIPE
            )
            return result.stdout.decode("utf-8")
        elif current_system == "Darwin":  # macOS
            # Use pbpaste to obtain secret data
            result = subprocess.run(["pbpaste"], stdout=subprocess.PIPE)
            return result.stdout.decode("utf-8")
        elif current_system == "Windows":
            # Use PowerShell to obtain secret data on Windows
            result = subprocess.run(
                ["powershell", "-command", "Get-Clipboard"], stdout=subprocess.PIPE
            )
            return result.stdout.decode("utf-8")
        else:
            print(f"Unsupported OS: {current_system}")
            return ""
    except subprocess.CalledProcessError:
        print("Failed to obtain secret data.")
        return ""


async def generate_link(token_id, lifetime="60s"):
    data = obtain_secret_data()
    base_url = "https://secretlink.net"
    evaluate_lifetime_in_seconds(lifetime)

    async def generate_token():
        url = f"{base_url}/{token_id}"
        payload = {"secret": data}
        try:
            response = await asyncio.to_thread(requests.post, url, data=payload)
            if response.status_code == 200:
                return True
            else:
                print(f"Failed to generate token: Status code {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Failed to generate token: {e}")
            return False

    success = await generate_token()
    if success:
        print(f"Token generated successfully: https://secretlink.net/{token_id}")
    else:
        print("Failed to generate the token.")


def translate_lifetime_to_seconds(lifetime):
    parts = lifetime.split()
    if len(parts) != 2:
        raise ValueError(
            "Lifetime must be in the format '<number> <unit>' (e.g., '1 day')."
        )

    value, unit = parts
    try:
        value = int(value)
    except ValueError:
        raise ValueError("The first part of lifetime must be an integer.")

    if "second" in unit or "seconds" in unit:
        return value
    elif "minute" in unit or "minutes" in unit:
        return value * 60
    elif "hour" in unit or "hours" in unit:
        return value * 3600
    elif "day" in unit or "days" in unit:
        return value * 86400
    elif "week" in unit or "weeks" in unit:
        return value * 604800
    else:
        raise ValueError(
            "Invalid time unit provided. Use seconds, minutes, hours, days, or weeks."
        )


if __name__ == "__main__":
    asyncio.run(
        generate_link("abc123", "This secret will self-destruct!", "1 day")
    )  # Example usage
