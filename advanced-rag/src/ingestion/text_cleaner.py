import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving meaningful content.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove spaces at the beginning/end of each line
    lines = [line.strip() for line in text.split("\n")]

    # Remove completely empty lines
    lines = [line for line in lines if line]

    # Rejoin lines
    text = "\n".join(lines)

    # Replace multiple consecutive spaces with a single space
    text = re.sub(r" {2,}", " ", text)

    return text.strip()