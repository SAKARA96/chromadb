import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection

# Initialize Chroma client with persistent storage (if needed)
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_store"  # Specify where to store data
))

def create_collection(collection_name: str):
    """Gets or creates a ChromaDB collection and returns the collection object."""
    return chroma_client.get_or_create_collection(name=collection_name)

async def add_to_collection(
    text: str,
    embedding: list[float],
    doc_id: str,
    collection: Collection = None
):
    if collection is None:
        collection = create_collection(collection_name="test")

    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[doc_id]
    )