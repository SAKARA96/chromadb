from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.document.batch import process_file,process_text,process_embeddings
from app.db.client import chroma_client,update_query_centroid,update_top_k_collections,update_top_k_documents
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
       file.filename: request.create_upload_document()  # Use the factory function for each file
        for file in files
    }

    logger.debug(f"File map created: {file_map}")

    # Use asyncio.gather to run file processing concurrently
    logger.info("Starting file processing...")
    await asyncio.gather(
        *[
            process_file(
                file=file,
                upload_document=file_map[file.filename]
            )
            for file in files
        ]
    )
    logger.info("File processing completed.")

    # Batch processing for converting to vector embeddings using sentence transformer
    logger.info("Starting batch processing for text embeddings...")
    await asyncio.gather(
        *[
            process_text(
                filename=filename,
                upload_document=file_map[filename]
            )
            for filename in file_map
        ]
    )
    logger.info("Text embeddings batch processing completed.")

    # Batch processing to add embeddings and documents to chromadb
    logger.info("Starting batch processing to add embeddings to chromadb...")
    await asyncio.gather(
        *[
            process_embeddings(
                filename=filename,
                upload_document=file_map[filename]
            )
            for filename in file_map
        ]
    )
    logger.info("Embeddings and documents added to chromadb.")

    logger.info("File upload and processing completed.")
    
    return JSONResponse(
        content={
            "state": [upload_doc.to_dict() for filename, upload_doc in file_map.items()]
        }
    )

#---------------------------------------------------------------------------------------------------------------

@router.post("/search/")
async def search(requestParam: request.SearchRequest):

    queries = [requestParam.query]

    file_map = {
        idx: request.create_search_document(
            text=query
        )
        for idx, query in enumerate(queries)
    }

    #convert text to embeddings
    logger.info("Starting batch processing for query embeddings...")
    await asyncio.gather(
        *[
            process_text(
                filename=query,
                upload_document=file_map[query]
            )
            for query in file_map
        ]
    )
    logger.info("Query embeddings batch processing completed")

    #get centroid embedding mean from query input
    logger.info("Starting update_query_centroid for all queries...")
    await asyncio.gather(
        *[
            update_query_centroid(
                filename=query,
                document=file_map[query]
            )
            for query in file_map
        ]
    )
    logger.info("update_query_centroid for all queries completed")

    # #update_top_k_collections per query
    logger.info("Finding top k collections for all queries...")
    await asyncio.gather(
        *[
            update_top_k_collections(
                query=query,
                document=file_map[query],
                top_k=requestParam.top_k_collections
            )
            for query in file_map
        ]
    )

    logger.info("update_top_k_collections for all queries completed")  

    #update_top_k_documents per query
    logger.info("Finding update_top_k_documents for all queries...")
    await asyncio.gather(
        *[
            update_top_k_documents(
                query=query,
                document=file_map[query],
                top_k=requestParam.top_k_documents
            )
            for query in file_map
        ]
    )
    logger.info("update_top_k_documents for all queries completed")  
    
    
    return JSONResponse(
        content={
            "state": [upload_doc.to_dict() for filename, upload_doc in file_map.items()]
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
