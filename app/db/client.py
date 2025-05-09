import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid
import time
from app.logger import logger

# Initialize Chroma client with persistent storage (if needed)
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

def get_or_create_collection(collection_name: str):
    """Gets or creates a ChromaDB collection and returns the collection object."""
    return chroma_client.get_or_create_collection(name=collection_name)

#---------------------------------------------------------------------------------------------------------------

async def add_to_collection(
    text: str,
    embedding: list[float],
    doc_id: str,
    filename: str,
    collection: Collection = None
):
    # Log the function entry and input parameters
    logger.debug(f"add_to_collection called with doc_id: {doc_id}, filename: {filename}, collection: {collection}")

    if collection is None:
        logger.debug("No collection provided. Creating a new collection with the name 'test'.")
        collection = get_or_create_collection(collection_name="test")

    try:
        logger.debug(f"Adding document to collection: {collection.name}")
        
        # Add the document to the collection
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{"filename": filename}]
        )
        
        logger.debug(f"Successfully added document with doc_id: {doc_id} to collection: {collection.name}")

        # Update the collection centroid
        await update_collection_centroid(collection=collection)
        logger.debug(f"Collection centroid updated for collection: {collection.name}")

    except Exception as e:
        logger.error(f"An error occurred while adding the document with doc_id: {doc_id} to collection: {collection.name}. Error: {e}", exc_info=True)

#---------------------------------------------------------------------------------------------------------------

async def update_collection_centroid(collection: Collection):
    """
    Gets a collection object and updates the centroid embedding of the object
    """
    logger.debug(f"update_collection_centroid called for collection: {collection.name if collection else 'Unnamed'}")

    if collection is None:
        logger.debug("No collection provided. Creating a new collection with the name 'test'.")
        collection = get_or_create_collection(collection_name="test")

    try:
        # Fetch all documents from the collection
        logger.debug("Fetching all documents from the collection...")
        all_docs = collection.get(include=["embeddings"])
        ids = all_docs.get("ids", [])
        embeddings = all_docs.get("embeddings", [])

        logger.debug(f"Fetched {len(ids)} documents from the collection.")

        # Filter out the embeddings for the centroid calculation
        filtered_embeddings = [
            emb for doc_id, emb in zip(ids, embeddings) if doc_id != "centroid"
        ]
        logger.debug(f"Filtered out centroid embedding. {len(filtered_embeddings)} embeddings will be used to calculate the centroid.")

        # Calculate the centroid
        logger.debug(f"Calculating centroid from {len(filtered_embeddings)} embeddings...")
        centroid = np.mean(filtered_embeddings, axis=0)
        logger.debug("Centroid calculated successfully.")

        centroid_metadata = {
            "description": "Centroid embedding of the collection"
        }

        # Add the centroid embedding as a new document
        logger.debug(f"Adding centroid document to the collection with id 'centroid'.")
        collection.add(
            documents=["Centroid Document"],
            embeddings=[centroid],
            metadatas=[centroid_metadata],
            ids=["centroid"]
        )

        logger.debug(f"Centroid document added successfully to collection: {collection.name}")

    except Exception as e:
        logger.error(f"An error occurred while updating the centroid for collection: {collection.name if collection else 'Unnamed'}. Error: {str(e)}", exc_info=True)

#---------------------------------------------------------------------------------------------------------------

async def top_1_collection(query_embedding: np.ndarray, threshold: float = 0.65) -> str:
    client = chroma_client

    logger.debug(f"top_1_collection called with threshold: {threshold}")

    collection_names = client.list_collections()
    logger.debug(f"Found {len(collection_names)} collections.")

    # If no collections are present, create a new collection and return it
    if not collection_names:
        new_collection_name = f"collection-{uuid.uuid4()}"
        logger.debug(f"No collections found. Creating a new collection: {new_collection_name}")
        get_or_create_collection(new_collection_name)
        return new_collection_name
    
    best_collection = None
    highest_similarity = -1

    # Iterate through each collection to retrieve the centroid and compute similarity
    for collection_name in collection_names:
        logger.debug(f"Processing collection: {collection_name}")
        collection = collection_name
        
        try:
            # Retrieve the centroid of the collection
            logger.debug(f"Fetching centroid for collection: {collection_name}")
            centroid_doc = collection.get(ids=["centroid"], include=["embeddings"])

            if not centroid_doc:
                logger.debug(f"Centroid not found for collection {collection_name}. Updating centroid.")
                await update_collection_centroid(collection=collection)
                centroid_doc = collection.get(ids=["centroid"], include=["embeddings"])

            centroid_embedding = np.array(centroid_doc['embeddings'][0])
            similarity_score = cosine_similarity([query_embedding], [centroid_embedding])[0][0]
            logger.debug(f"Similarity score for collection {collection_name}: {similarity_score}")

            if similarity_score > highest_similarity:
                highest_similarity = similarity_score
                best_collection = collection.name
                logger.debug(f"New best collection found: {best_collection} with similarity {highest_similarity}")

        except Exception as e:
            logger.error(f"Error retrieving centroid for collection '{collection_name}': {e}", exc_info=True)
            continue  # In case a collection doesn't have a centroid or some other issue

    if best_collection and highest_similarity >= threshold:
        logger.debug(f"Best collection found: {best_collection} with similarity: {highest_similarity}")
        return best_collection
    else:
        new_collection_name = f"collection-{uuid.uuid4()}"
        logger.debug(f"No matching collection found or similarity below threshold. Creating a new collection: {new_collection_name}")
        get_or_create_collection(new_collection_name)
        return new_collection_name

#---------------------------------------------------------------------------------------------------------------

async def update_query_centroid(filename:str,file_map:dict):
    """
    Gets a file_map object and updates the centroid embedding of the object
    """

    logger.debug(f"start update_query_centroid for query : {filename}")
    
    try:
        query_object = file_map[filename]
        if query_object:
            embeddings = query_object["embedding"]["content"]
            centroid = np.mean(embeddings,axis=0)
            file_map[filename]["centroid"] = {
                "content":centroid,
                "error":None
            }
        else:
            file_map[filename]["centroid"] = {
                "content":-1,
                "error":f"Error calculating centroid"
            }

    except Exception as e:
        logger.error(f"An error occurred while updating the centroid for query. Error: {str(e)}", exc_info=True)
    
    logger.debug(f"Completed update_query_centroid for query : {filename}")

#---------------------------------------------------------------------------------------------------------------