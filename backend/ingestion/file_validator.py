
from pathlib import Path

MAX_FILE_SIZE = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
    ".csv",
    ".xlsx",
    ".json",
    ".md"
}


class FileValidationError(Exception):
    pass


def validate_file(filename, file_bytes):

    if not filename:
        raise FileValidationError("Filename is required")

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file format: {extension}"
        )

    if not file_bytes:
        raise FileValidationError("Uploaded file is empty")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise FileValidationError(
            "File size exceeds the 20 MB limit"
        )

    if extension == ".pdf":
        if not file_bytes.startswith(b"%PDF-"):
            raise FileValidationError(
                "Invalid PDF file signature"
            )

    if extension in {".docx", ".xlsx"}:
        if not file_bytes.startswith(b"PK"):
            raise FileValidationError(
                "Invalid Office document signature"
            )

    return {
        "filename": Path(filename).name,
        "extension": extension,
        "size_bytes": len(file_bytes),
        "status": "valid"
    }
