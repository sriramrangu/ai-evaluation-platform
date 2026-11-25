
import re
import unicodedata


def clean_text(text):

    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.replace("\x00", "")

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()
