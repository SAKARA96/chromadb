from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    top_k_collections: int = 5
    top_k_documents: int = 5