
from pydantic import BaseModel, Field
from typing import Any
from uuid import uuid4


class ProcessedDocument(BaseModel):

    document_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    source: str

    file_type: str

    content: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )