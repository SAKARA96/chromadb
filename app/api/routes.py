from fastapi import APIRouter, UploadFile, File, HTTPException
from app.document.extract import extract_text
from app.db.client import chroma_client

router = APIRouter()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Extract the text using the extract_text function
        text = extract_text(file.file, file.filename)

        return {
            "filename": file.filename,
            "total_characters": len(text),
            "text": text
        }

    except ValueError as ve:
        raise HTTPException(status_code=415, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    
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
