from dataclasses import dataclass,field
from typing import Any,List,Optional
from enum import Enum
import uuid
import torch 


def tensor_to_list(tensors: Optional[List[torch.Tensor]]) -> List:
    if not tensors:
        return []
    return [t.tolist() for t in tensors]

@dataclass
class Text:
    content: List[str] = field(default_factory=list)
    error : Optional[str] = None
    shape: List[int] = field(default_factory=list)

    def to_dict(self):
        return {
            "content": self.content,
            "error": self.error,
            "shape": self.shape
        }

@dataclass
class Embedding:
    content: Optional[List[torch.Tensor]] = field(default_factory=list)
    vectordb_embeddings: Optional[List[float]] = field(default_factory=list)
    error: Optional[str] = None
    shape: List[int] = field(default_factory=list)

    def to_dict(self):
        return {
            "content": tensor_to_list(self.content),
            "error": self.error,
            "shape": self.shape
        }

    def convert_to_list_floats(self):
        self.vectordb_embeddings = []
        for embed in self.content:
            self.vectordb_embeddings.extend(embed.tolist())

@dataclass
class BaseDocument:
    text: Text
    embedding: Embedding
    uuid : str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self):
        return {
            "text": self.text.to_dict(),
            "embedding": self.embedding.to_dict(),
            "uuid": self.uuid
        }

class StatusEnum(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    INIT = "default"

@dataclass
class Status:
    code: Optional[StatusEnum] = None
    error: Optional[str] = None

    def to_dict(self):
        return {
            "status": self.code.value if self.code else None,
            "error": self.error
        }

@dataclass
class UploadDocument(BaseDocument):
    status: Optional[Status] = None
    collection: Optional[str] = None

    def to_dict(self):
        data = super().to_dict()
        data.update({
            "status": self.status.to_dict() if self.status else None,
            "collection": self.collection
        })
        return data
    
@dataclass
class SearchDocument(BaseDocument):
    centroid: Optional[Embedding] = None
    top_k_collections: List[str] = field(default_factory=list)
    top_k_results: List[str] = field(default_factory=list)

@dataclass
class SearchRequest:
    query: str
    top_k_collections: int = 5
    top_k_documents: int = 5

def create_upload_document() -> UploadDocument:
    text_data = Text(content=[""], error=None, shape=[])
    embedding_data = Embedding(content=[torch.empty(0)], error=None, shape=[])
    
    # Create the UploadDocument instance
    return UploadDocument(
        text=text_data,
        embedding=embedding_data,
        uuid=str(uuid.uuid4()), 
        status=Status(code=StatusEnum.INIT, error=None),
        collection=None,
    )