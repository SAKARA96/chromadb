import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid
import time

# Initialize Chroma client with persistent storage (if needed)
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

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

    try:
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas = [{"filename":filename}]
        )
        await update_collection_centroid(collection=collection)

    except Exception as e:
        print(f"An error occurred while adding the document: {e}")

async def update_collection_centroid(collection: Collection):
    """
    Gets a collection object and updates the centroid embedding of the object
    """
    if collection is None:
        collection = get_or_create_collection(collection_name="test")

    # Fetch all documents
    all_docs = collection.get(include=["embeddings"])
    ids = all_docs.get("ids", [])
    embeddings = all_docs.get("embeddings", [])

    filtered_embeddings = [
        emb for doc_id, emb in zip(ids, embeddings) if doc_id != "centroid"
    ]
    #Calculate centroid 
    centroid = np.mean(filtered_embeddings, axis=0)
    centroid_metadata = {
        "description": "Centroid embedding of the collection"
    }
    collection.add(
        documents=["Centroid Document"],
        embeddings=[centroid],  
        metadatas=[centroid_metadata],
        ids=["centroid"]  
    )

async def top_1_collection(query_embedding: np.ndarray,threshold: float = 0.65) -> str:
    client = chroma_client

    collection_names = client.list_collections()
    
    #if no collections present create a new collection and return it
    if not collection_names:
        new_collection_name = f"collection-{uuid.uuid4()}"
        print("creating a new collection",new_collection_name)
        get_or_create_collection(new_collection_name)
        return new_collection_name
    
    best_collection = None

    highest_similarity = -1

    # Iterate through each collection to retrieve the centroid and compute similarity
    for collection_name in collection_names:
        # collection = client.get_collection(collection_name)
        collection = collection_name
        try:
            centroid_doc = collection.get(ids=["centroid"],include=["embeddings"])
            if not centroid_doc:
                await update_collection_centroid(collection=collection)
                centroid_doc = collection.get(
                    ids=["centroid"],
                    include=["embeddings"]
                )
            centroid_embedding = np.array(centroid_doc['embeddings'][0])
            similarity_score = cosine_similarity([query_embedding], [centroid_embedding])[0][0]
            if similarity_score > highest_similarity:
                highest_similarity = similarity_score
                best_collection = collection.name

        except Exception as e:
            print(f"Error retrieving centroid for collection '{collection}': {e}")
            continue  # In case a collection doesn't have a centroid or some other issue

    if best_collection and highest_similarity >= threshold:
        return best_collection
    else:
        new_collection_name = f"collection-{uuid.uuid4()}"
        get_or_create_collection(new_collection_name)
        return new_collection_name