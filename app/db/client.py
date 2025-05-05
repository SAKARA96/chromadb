import chromadb
from chromadb.config import Settings

# Initialize Chroma client with persistent storage (if needed)
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_store"  # Specify where to store data
))

def create_collection(collection_name: str):
    """Creates a collection if it doesn't exist."""
    if collection_name not in chroma_client.list_collections():
        chroma_client.create_collection(collection_name)
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")