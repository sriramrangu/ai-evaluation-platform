
import json
import sqlite3

from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone


DEFAULT_DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "document_store.db"
)


class DocumentStore:

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

        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

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
                CREATE TABLE IF NOT EXISTS stored_documents (
                    document_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,

                    FOREIGN KEY (document_id)
                    REFERENCES stored_documents(document_id)
                    ON DELETE CASCADE,

                    UNIQUE(document_id, chunk_index)
                )
            """)

    def save_document(self, document, chunks):

        created_at = datetime.now(timezone.utc).isoformat()

        with self.get_connection() as connection:

            connection.execute(
                """
                INSERT INTO stored_documents (
                    document_id,
                    source,
                    file_type,
                    content,
                    metadata,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    document.source,
                    document.file_type,
                    document.content,
                    json.dumps(document.metadata),
                    created_at
                )
            )

            for chunk in chunks:

                connection.execute(
                    """
                    INSERT INTO document_chunks (
                        chunk_id,
                        document_id,
                        chunk_index,
                        content,
                        metadata
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["chunk_id"],
                        chunk["document_id"],
                        chunk["chunk_index"],
                        chunk["content"],
                        json.dumps(chunk["metadata"])
                    )
                )

        return {
            "status": "stored",
            "document_id": document.document_id,
            "total_chunks": len(chunks)
        }

    def get_document(self, document_id):

        with self.get_connection() as connection:

            row = connection.execute(
                """
                SELECT *
                FROM stored_documents
                WHERE document_id = ?
                """,
                (document_id,)
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["metadata"] = json.loads(result["metadata"])

        return result

    def get_chunks(self, document_id):

        with self.get_connection() as connection:

            rows = connection.execute(
                """
                SELECT *
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index
                """,
                (document_id,)
            ).fetchall()

        results = []

        for row in rows:

            chunk = dict(row)
            chunk["metadata"] = json.loads(chunk["metadata"])

            results.append(chunk)

        return results

    def list_documents(self):

        with self.get_connection() as connection:

            rows = connection.execute(
                """
                SELECT
                    document_id,
                    source,
                    file_type,
                    created_at
                FROM stored_documents
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def delete_document(self, document_id):

        with self.get_connection() as connection:

            cursor = connection.execute(
                """
                DELETE FROM stored_documents
                WHERE document_id = ?
                """,
                (document_id,)
            )

        return cursor.rowcount > 0
