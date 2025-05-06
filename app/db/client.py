import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
import numpy as np

# Initialize Chroma client with persistent storage (if needed)
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_store"  # Specify where to store data
))

def get_or_create_collection(collection_name: str):
    """Gets or creates a ChromaDB collection and returns the collection object."""
    return chroma_client.get_or_create_collection(name=collection_name)

async def add_to_collection(
    text: str,
    embedding: list[float],
    doc_id: str,
    filename: str,
    collection: Collection = None
):
    if collection is None:
        collection = get_or_create_collection(collection_name="test")

    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[doc_id],
        metadatas = [{"filename":filename}]
    )

def update_collection_centroid(collection: Collection):
    """
    Gets a collection object and updates the centroid embedding of the object
    """
    if collection is None:
        collection = get_or_create_collection(collection_name="test")

    # Fetch all documents except the centroid
    document_ids = collection.get_ids()  # Get all IDs
    document_ids = [doc_id for doc_id in document_ids if doc_id != "centroid"]  # Exclude the centroid

    #Get all embeddings
    embeddings = collection.get_embeddings(ids=document_ids)

    #Calculate centroid 
    centroid = np.mean(embeddings, axis=0)

    centroid_metadata = {
        "centroid_embedding": centroid.tolist(),
        "description": "Centroid embedding of the collection"
    }

    collection.add(
        documents=["Centroid Document"],
        embeddings=[centroid],  
        metadatas=[centroid_metadata],
        ids=["centroid"]  
    )