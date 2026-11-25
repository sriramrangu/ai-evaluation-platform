
import hashlib
import sqlite3

from pathlib import Path
from datetime import datetime, timezone

from contextlib import contextmanager


DEFAULT_DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "document_registry.db"
)


class DocumentRegistry:

    def __init__(self, db_path=DEFAULT_DB_PATH):

        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.initialize()

    
    @contextmanager
    def get_connection(self):

        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30
        )

        try:
            yield connection
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()


    def initialize(self):

        with self.get_connection() as connection:

            connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
            """)

    @staticmethod
    def calculate_hash(file_bytes):

        return hashlib.sha256(
            file_bytes
        ).hexdigest()

    def find_duplicate(self, file_bytes):

        file_hash = self.calculate_hash(file_bytes)

        with self.get_connection() as connection:

            row = connection.execute(
                """
                SELECT document_id, source, file_type, created_at
                FROM documents
                WHERE file_hash = ?
                """,
                (file_hash,)
            ).fetchone()

        if row is None:
            return None

        return {
            "document_id": row[0],
            "source": row[1],
            "file_type": row[2],
            "created_at": row[3]
        }

    def register_document(self, document, file_bytes):

        file_hash = self.calculate_hash(file_bytes)

        created_at = datetime.now(
            timezone.utc
        ).isoformat()

        with self.get_connection() as connection:

            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO documents (
                    document_id,
                    source,
                    file_type,
                    file_hash,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    document.source,
                    document.file_type,
                    file_hash,
                    created_at
                )
            )

            is_new = cursor.rowcount == 1

            row = connection.execute(
                """
                SELECT document_id, source, file_type, created_at
                FROM documents
                WHERE file_hash = ?
                """,
                (file_hash,)
            ).fetchone()

        return {
            "status": "registered" if is_new else "duplicate",
            "is_duplicate": not is_new,
            "document_id": row[0],
            "source": row[1],
            "file_type": row[2],
            "created_at": row[3],
            "file_hash": file_hash
        }
