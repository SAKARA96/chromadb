from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.document.batch import process_file,process_text,process_embeddings
from app.db.client import chroma_client,update_query_centroid,update_top_k_collections
import app.api.request as request
from app.logger import logger
import asyncio
from typing import List
import uuid

router = APIRouter()

#---------------------------------------------------------------------------------------------------------------

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    logger.info(f"Starting file upload for {len(files)} files.")

    file_map = {
        file.filename: {
            "text": {},
            "embedding": {},
            "uuid": str(uuid.uuid4())
        }
        for file in files
    }

    logger.debug(f"File map created: {file_map}")

    # Use asyncio.gather to run file processing concurrently
    logger.info("Starting file processing...")
    await asyncio.gather(*[process_file(file=file, file_map=file_map) for file in files])
    logger.info("File processing completed.")

    # Batch processing for converting to vector embeddings using sentence transformer
    logger.info("Starting batch processing for text embeddings...")
    await asyncio.gather(*[process_text(filename=filename,file_map= file_map) for filename in file_map])
    logger.info("Text embeddings batch processing completed.")

    # Batch processing to add embeddings and documents to chromadb
    logger.info("Starting batch processing to add embeddings to chromadb...")
    await asyncio.gather(*[process_embeddings(filename=filename,file_map= file_map)for filename in file_map])
    logger.info("Embeddings and documents added to chromadb.")

    logger.info("File upload and processing completed.")

    return JSONResponse(
        content={
            "state":file_map
        }
    )

#---------------------------------------------------------------------------------------------------------------

@router.post("/search/")
async def search(request: request.SearchRequest):
    
    file_map = {
        "query":{
            "text":{"content":request.query, "total_characters":len(request.query),"error":None},
            "embedding":{},
            "uuid":str(uuid.uuid4())
        }
    }

    #convert text to embeddings
    logger.info("Starting batch processing for query embeddings...")
    await asyncio.gather(*[process_text(filename=filename,file_map=file_map)for filename in file_map])
    logger.info("Query embeddings batch processing completed")

    #get centroid embedding mean from query input
    logger.info("Starting update_query_centroid for all queries...")
    await asyncio.gather(*[update_query_centroid(query=query,file_map=file_map)for query in file_map])
    logger.info("update_query_centroid for all queries completed")

    #update_top_k_collections per query
    logger.info("Finding top k collections for all queries...")
    await asyncio.gather(*[update_top_k_collections(query=query,file_map=file_map,top_k=request.top_k_collections)for query in file_map])
    logger.info("update_top_k_collections for all queries completed")  
    
    return JSONResponse(
        content={
            "state":file_map
        }
    )

#---------------------------------------------------------------------------------------------------------------

@router.get("/health_db")
async def check_chroma():
    try:
        collections = chroma_client.list_collections()
        collection_names = [collection.name for collection in collections]
    
        if "test" in collection_names:
            return {"status": "ChromaDB is up", "collection": "test"}
        else:
            return {"status": "ChromaDB is up", "collection": "test not found"}
    except Exception as e:
        return {"status": "Error", "message": str(e)}
