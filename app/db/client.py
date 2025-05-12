import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid
import time
from app.logger import logger
import torch 
from typing import List

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

    if collection == None:
        logger.debug("No collection provided. Creating a new collection with the name 'test'.")
        collection = get_or_create_collection(collection_name="test")

    try:
        logger.debug(f"Adding document to collection: {collection.name}")
        
        # Add the document to the collection
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{"filename": filename, "is_centroid":False}]
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

    try:
        # Fetch all documents from the collection
        logger.debug("Fetching all documents from the collection...")
        all_docs = collection.get(include=["embeddings"])
        ids = all_docs.get("ids", [])
        embeddings = all_docs.get("embeddings", [])

        logger.debug(f"Fetched {len(ids)} documents from the collection.")

        # Filter out the embeddings for the centroid calculation
        filtered_embeddings = [
            torch.Tensor(emb) for doc_id, emb in zip(ids, embeddings) if doc_id != "centroid"
        ]
        logger.debug(f"Filtered out centroid embedding. {len(filtered_embeddings)} embeddings will be used to calculate the centroid.")

        # Calculate the centroid
        logger.debug(f"Calculating centroid from {len(filtered_embeddings)} embeddings...")
        all_embeddings = np.vstack(filtered_embeddings)
        centroid = np.mean(all_embeddings, axis=0)
        logger.debug("Centroid calculated successfully.")

        centroid_metadata = {
            "description": "Centroid embedding of the collection",
            "is_centroid": True
        }

        # Add the centroid embedding as a new document
        logger.debug(f"Adding centroid document to the collection with id 'centroid'.")
        collection.add(
            documents=["Centroid Document"],
            embeddings=[centroid.tolist()],
            metadatas=[centroid_metadata],
            ids=["centroid"]
        )

        logger.debug(f"Centroid document added successfully to collection: {collection.name}")

    except Exception as e:
        logger.error(f"An error occurred while updating the centroid for collection: {collection.name if collection else 'Unnamed'}. Error: {str(e)}", exc_info=True)

#---------------------------------------------------------------------------------------------------------------

async def top_1_collection(query_embedding: List[torch.Tensor], threshold: float = 0.35) -> str:
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

    stacked_embeddings = torch.stack(query_embedding)
    avg_query_embedding = torch.mean(stacked_embeddings, dim=0)
    
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


            centroid_embedding = torch.Tensor(centroid_doc['embeddings'][0])  
            similarity_score = torch.nn.functional.cosine_similarity(
                avg_query_embedding.unsqueeze(0), 
                centroid_embedding.unsqueeze(0), 
                dim=1
            ).item()
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

async def update_query_centroid(query:str,file_map:dict ):
    """
    Gets a file_map object and updates the centroid embedding of the object
    """

    logger.debug(f"start update_query_centroid for query : {query}")
    
    try:
        query_object = file_map[query]
        if query_object:
            embeddings = torch.Tensor(query_object["embedding"]["content"])
            all_embeddings = np.vstack(embeddings)
            centroid = np.mean(all_embeddings, axis=0)
            file_map[query]["centroid"] = {
                "content":centroid.tolist(),
                "error":None
            }
        else:
            file_map[query]["centroid"] = {
                "content":-1,
                "error":f"Error calculating centroid"
            }

    except Exception as e:
        logger.error(f"An error occurred while updating the centroid for query. Error: {str(e)}", exc_info=True)
    
    logger.debug(f"Completed update_query_centroid for query : {query}")

#---------------------------------------------------------------------------------------------------------------

async def update_top_k_collections(query:str, file_map:dict, top_k:int, threshold: float = 0.0):
    """
    Gets a file_map and updates the top_k_collections for every query 
    """

    logger.debug(f"start update_top_k_collections for query : {query}")

    try:
        query_object = file_map[query]
        if query_object:
            if not query_object["centroid"]["error"]:
                query_centroid = torch.Tensor(file_map[query]["centroid"]["content"])
                collections_list = chroma_client.list_collections()

                similarity_scores = []

                for collection in collections_list:
                    logger.debug(f"Fetching centroid for collection: {collection.name}")
                    centroid_doc = collection.get(ids=["centroid"], include=["embeddings"])

                    if len(centroid_doc['embeddings']) == 0:
                        logger.warning(f"No centroid embedding found for collection: {collection.name}")
                        continue

                    centroid_embedding = torch.Tensor(centroid_doc['embeddings'][0])  
                    similarity_score = torch.nn.functional.cosine_similarity(
                        query_centroid.unsqueeze(0), 
                        centroid_embedding.unsqueeze(0), 
                        dim=1
                    ).item()

                    similarity_scores.append((collection.name, similarity_score))

                sorted_collections = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

                file_map[query]["top_k_collections"]={
                    "content":sorted_collections,
                    "error": None
                }  

            else:
                file_map[query]["top_k_collections"] = {
                "content":[],
                "error": f"Error getting centroid for query"
            }    
        else:
            file_map[query]["top_k_collections"] = {
                "content":[],
                "error": f"Error finding top_k_collections"
            }

    except Exception as e:
        logger.error(f"An error occuered while finding top_k_collections for query. Error: {str(e)}",exc_info=True)

    logger.debug(f"Completed update_top_k_collections for query : {query}")

#---------------------------------------------------------------------------------------------------------------

async def update_top_k_documents(query:str,file_map:dict,top_k:int):
    """
    Gets a file_map and query and tries to find relevant documents from a collection to send as additional context
    """

    logger.debug("start update_top_k_documents for query : {query}")

    try:
        query_object = file_map[query]
        if query_object:
            query_embedding = query_object["embedding"]["content"]
            collection_list = query_object["top_k_collections"]["content"]
            file_map[query]["top_k_documents"] = {
                "content":[],
                "error":None
            }
            for collection_obj in collection_list:
                collection_name = collection_obj[0]
                logger.info(f"collection_name {collection_name}")
                collection = chroma_client.get_collection(name=collection_name)
                
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=top_k,include=["documents"],
                    where={"is_centroid": False}
                )

                file_map[query]["top_k_documents"]["content"].extend(results["documents"][0])
        else:
            file_map[query]["top_k_documents"] = {
                "content":[],
                "error": f"Error finding top_k_documents"
            }

    
    except Exception as e:
        logger.error(f"An error occuered while finding top_k_documents for query. Error : {str(e)}",exc_info=True)
    
    logger.debug(f"Completed update_top_k_documents for query : {query}")