from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.document.batch import process_file,process_text,process_embeddings
from app.db.client import chroma_client,get_or_create_collection,update_collection_centroid
import asyncio
from typing import List
import uuid

router = APIRouter()

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):

    file_map = {
        file.filename: {
            "text": {},
            "embedding": {},
            "uuid": str(uuid.uuid4())
        }
        for file in files
    }

    # Use asyncio.gather to run file processing concurrently
    await asyncio.gather(*[process_file(file=file, file_map=file_map) for file in files])

    # Batch processing for converting to vector embeddings using sentence transformer
    await asyncio.gather(*[process_text(filename=filename,file_map= file_map) for filename in file_map])

    # Batch processing to add embeddings and documents to chromadb
    await asyncio.gather(*[process_embeddings(filename=filename,file_map= file_map)for filename in file_map])

    return JSONResponse(
        content={
            "state":file_map
        }
    )

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
