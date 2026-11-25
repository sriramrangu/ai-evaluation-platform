
import io
import json

import pymupdf
import pandas as pd

from docx import Document as WordDocument

from backend.ingestion.file_validator import validate_file
from backend.ingestion.schemas import ProcessedDocument


class DocumentProcessingError(Exception):
    pass


def extract_pdf(file_bytes):

    try:
        pages = []
        pdf = pymupdf.open(
            stream=file_bytes,
            filetype="pdf"
            )

        if pdf.is_encrypted:
            raise DocumentProcessingError(
                "Encrypted PDFs are not supported"
            )

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text")

            if text.strip():
                pages.append(
                    f"[Page {page_number}]\n{text}"
                )

        page_count = len(pdf)
        pdf.close()

        content = "\n\n".join(pages)

        if not content.strip():
            raise DocumentProcessingError(
                "No extractable text found. "
                "This PDF may require OCR."
            )

        return content, {
            "page_count": page_count
        }

    except DocumentProcessingError:
        raise

    except Exception as error:
        raise DocumentProcessingError(
            f"PDF extraction failed: {error}"
        )


def extract_txt(file_bytes):

    for encoding in ["utf-8-sig", "utf-8", "cp1252"]:

        try:
            return file_bytes.decode(encoding), {
                "encoding": encoding
            }

        except UnicodeDecodeError:
            continue

    raise DocumentProcessingError(
        "Unable to decode text file"
    )


def extract_docx(file_bytes):

    try:
        document = WordDocument(
            io.BytesIO(file_bytes)
        )

        content = []

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text)

        for table_number, table in enumerate(
            document.tables, start=1
        ):

            content.append(
                f"\n[Table {table_number}]"
            )

            for row in table.rows:
                values = [
                    cell.text.strip()
                    for cell in row.cells
                ]

                content.append(" | ".join(values))

        text = "\n".join(content)

        if not text.strip():
            raise DocumentProcessingError(
                "No extractable text found in DOCX"
            )

        return text, {
            "paragraph_count": len(document.paragraphs),
            "table_count": len(document.tables)
        }

    except DocumentProcessingError:
        raise

    except Exception as error:
        raise DocumentProcessingError(
            f"DOCX extraction failed: {error}"
        )


def extract_csv(file_bytes):

    try:
        dataframe = None

        for encoding in ["utf-8-sig", "utf-8", "cp1252"]:

            try:
                dataframe = pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding=encoding
                )
                break

            except UnicodeDecodeError:
                continue

        if dataframe is None:
            raise DocumentProcessingError(
                "Unable to decode CSV file"
            )

        if dataframe.empty:
            raise DocumentProcessingError(
                "CSV file contains no records"
            )

        content = dataframe.to_string(index=False)

        return content, {
            "row_count": len(dataframe),
            "column_count": len(dataframe.columns),
            "columns": dataframe.columns.tolist()
        }

    except DocumentProcessingError:
        raise

    except Exception as error:
        raise DocumentProcessingError(
            f"CSV extraction failed: {error}"
        )


def extract_xlsx(file_bytes):

    try:
        sheets = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=None
        )

        content = []
        sheet_metadata = {}

        for sheet_name, dataframe in sheets.items():

            if dataframe.empty:
                continue

            content.append(
                f"\n[Sheet: {sheet_name}]"
            )

            content.append(
                dataframe.to_string(index=False)
            )

            sheet_metadata[sheet_name] = {
                "rows": len(dataframe),
                "columns": dataframe.columns.tolist()
            }

        if not content:
            raise DocumentProcessingError(
                "Excel workbook contains no records"
            )

        return "\n".join(content), {
            "sheets": sheet_metadata
        }

    except DocumentProcessingError:
        raise

    except Exception as error:
        raise DocumentProcessingError(
            f"Excel extraction failed: {error}"
        )


def extract_json(file_bytes):

    try:
        data = json.loads(
            file_bytes.decode("utf-8-sig")
        )

        content = json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        )

        return content, {
            "json_type": type(data).__name__
        }

    except Exception as error:
        raise DocumentProcessingError(
            f"JSON extraction failed: {error}"
        )


def extract_markdown(file_bytes):

    content, metadata = extract_txt(file_bytes)

    return content, metadata


def load_document(filename, file_bytes):

    validation = validate_file(
        filename,
        file_bytes
    )

    extension = validation["extension"]

    extractors = {
        ".pdf": extract_pdf,
        ".txt": extract_txt,
        ".docx": extract_docx,
        ".csv": extract_csv,
        ".xlsx": extract_xlsx,
        ".json": extract_json,
        ".md": extract_markdown
    }

    try:
        content, extraction_metadata = extractors[
            extension
        ](file_bytes)

        return ProcessedDocument(
            source=validation["filename"],
            file_type=extension.lstrip("."),
            content=content,
            metadata={
                "file_size_bytes": validation["size_bytes"],
                **extraction_metadata
            }
        )

    except DocumentProcessingError:
        raise

    except Exception as error:
        raise DocumentProcessingError(
            f"Document processing failed: {error}"
        )
