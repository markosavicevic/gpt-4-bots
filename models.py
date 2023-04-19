from pydantic import BaseModel
from typing import Optional

class Metadata(BaseModel):
    title: str
    part: Optional[str] = None
    chapter: Optional[int] = None
    source: str
    chunk: int

class Record(BaseModel):
    id: str
    text: str
    metadata: Metadata