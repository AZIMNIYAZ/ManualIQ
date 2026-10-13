from pydantic import BaseModel, Field


class ManifestEntry(BaseModel):
    """One entry in documents/manifest.json."""

    file: str
    document_id: str = Field(pattern=r"^[a-z0-9_]+$")
    brand: str = Field(pattern=r"^[a-z0-9_]+$")
    model: str
    source_file: str | None = None  # original download name, for reference only


class Manifest(BaseModel):
    documents: list[ManifestEntry]


class Page(BaseModel):
    page_number: int  # 1-based, as printed in a PDF viewer
    text: str


class Document(BaseModel):
    document_id: str
    brand: str
    model: str
    source: str  # file name in documents/, shown in citations
    pages: list[Page]

    @property
    def text_length(self) -> int:
        return sum(len(p.text) for p in self.pages)
