import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from sklearn.metrics.pairwise import cosine_similarity
from app.api.request import SearchDocument
from app.document.extract import device
import numpy as np
import uuid
import time
from app.logger import logger
import torch 
from typing import List

# Initialize Chroma client with persistent storage (if needed)
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

def get_or_create_collection(collection_name: str):
    return chroma_client.get_or_create_collection(name=collection_name)

def generate_doc_ids(filename:str, num_chunks:int, start_idx:int):
    return [f"{filename}_chunk_{start_idx + idx}" for idx in range(num_chunks)]

def generate_metadatas(filename: str, num_chunks: int, start_idx: int , is_centroid: bool = False):
    return  [
                {
                    "filename": f"{filename}_chunk_{start_idx + idx}", 
                    "is_centroid": is_centroid
                } 
                for idx in range(num_chunks)
            ]


#---------------------------------------------------------------------------------------------------------------

async def add_to_collection(
    text: List[str],
    embedding: List[List[float]],
    filename: str,
    start_idx: int,
    collection: Collection = None
):
    # Log the function entry and input parameters
    logger.debug(f"add_to_collection called with start_idx: {start_idx}, filename: {filename}, collection: {collection}")

    try:
        logger.debug(f"Adding document to collection: {collection.name}")
        # Add the document to the collection
        collection.add(
            documents=text,
            embeddings=embedding,
            ids=generate_doc_ids(filename=filename,num_chunks=len(text),start_idx=start_idx),
            metadatas=generate_metadatas(filename=filename,num_chunks=len(text),start_idx=start_idx)
        )
        
        logger.debug(f"Successfully added documents with start_idx: {start_idx} to collection: {collection.name}")

        # Update the collection centroid
        await update_collection_centroid(collection=collection)
        logger.debug(f"Collection centroid updated for collection: {collection.name}")

    except Exception as e:
        logger.error(f"An error occurred while adding the document with start_idx: {start_idx} to collection: {collection.name}. Error: {e}", exc_info=True)

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
        stacked_embeddings = torch.stack(filtered_embeddings)
        avg_query_embedding = torch.mean(stacked_embeddings, dim=0)
        logger.debug("Centroid calculated successfully.")

        centroid_metadata = {
            "description": "Centroid embedding of the collection",
            "is_centroid": True
        }

        # Add the centroid embedding as a new document
        logger.debug(f"Adding centroid document to the collection with id 'centroid'.")
        collection.add(
            documents=["Centroid Document"],
            embeddings=[avg_query_embedding.tolist()],
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
    avg_query_embedding = torch.mean(stacked_embeddings, dim=0).to(device=device)
    
    best_collection = None
    highest_similarity = threshold

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


            centroid_embedding = torch.Tensor(centroid_doc['embeddings'][0]).to(device=device)
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

async def update_query_centroid(filename:str, document:SearchDocument ):
    """
    Gets a file_map object and updates the centroid embedding of the object
    """

    logger.debug(f"start update_query_centroid for query : {filename}")
    
    try:
        
        embeddings = document.embedding.content
        stacked_embeddings = torch.stack(embeddings)
        centroid = torch.mean(stacked_embeddings, dim=0)

        document.centroid.content = [centroid]
        document.centroid.error = None

    except Exception as e:
        logger.error(f"An error occurred while updating the centroid for query. Error: {str(e)}", exc_info=True)
    
    logger.debug(f"Completed update_query_centroid for query : {filename}")

#---------------------------------------------------------------------------------------------------------------

async def update_top_k_collections(query:str, document:SearchDocument, top_k:int, threshold: float = 0.25):
    """
    Gets a file_map and updates the top_k_collections for every query 
    """

    logger.debug(f"start update_top_k_collections for query : {query}")

    try:
        if not document.centroid.error:
            query_centroid = document.centroid.content
            stacked_embeddings = torch.stack(query_centroid)
            avg_query_embedding = torch.mean(stacked_embeddings, dim=0).to(device=device)
            
            collections_list = chroma_client.list_collections()
            similarity_scores = []

            for collection in collections_list:
                logger.debug(f"Fetching centroid for collection: {collection.name}")
                centroid_doc = collection.get(ids=["centroid"], include=["embeddings"])
                if len(centroid_doc['embeddings']) == 0:
                    logger.warning(f"No centroid embedding found for collection: {collection.name}")
                    continue

                centroid_embedding = torch.Tensor(centroid_doc['embeddings'][0]).to(device=device)
                similarity_score = torch.nn.functional.cosine_similarity(
                    avg_query_embedding.unsqueeze(0), 
                    centroid_embedding.unsqueeze(0), 
                    dim=1
                ).item()

                if similarity_score >= threshold:
                    logger.info(f"collection_name:{collection.name} similarity_score:{similarity_score}")
                    similarity_scores.append((collection.name, similarity_score))

            sorted_collections = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            collection_names = [name for name, _ in sorted_collections[:top_k]]
            
            document.top_k_collections = collection_names
        
    except Exception as e:
        logger.error(f"An error occuered while finding top_k_collections for query. Error: {str(e)}",exc_info=True)

    logger.debug(f"Completed update_top_k_collections for query : {query}")

#---------------------------------------------------------------------------------------------------------------

async def update_top_k_documents(query:str, document:SearchDocument, top_k:int, threshold: float = 0.0):
    """
    Gets a file_map and query and tries to find relevant documents from a collection to send as additional context
    """

    logger.debug("start update_top_k_documents for query : {query}")

    try:
        query_embedding = document.embedding.vectordb_embeddings
        collection_list = document.top_k_collections
        for collection_name in collection_list:
            logger.debug(f"collection_name {collection_name}")
            collection = chroma_client.get_collection(name=collection_name)
            
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents"],
                where={"is_centroid": False}
            )

            if "documents" in results and results["documents"]:
                for doc_list in results["documents"]:
                    document.top_k_results.extend(doc_list)

    except Exception as e:
        logger.error(f"An error occuered while finding top_k_documents for query. Error : {str(e)}",exc_info=True)
    
    logger.debug(f"Completed update_top_k_documents for query : {query}")