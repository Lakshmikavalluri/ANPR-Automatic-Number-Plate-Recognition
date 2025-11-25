import re

def clean_text(text):
    """
    Clean the OCR text by removing non-alphanumeric characters
    and converting to uppercase (typical for license plates).
    """
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    return text

