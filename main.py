from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.db.client import chroma_client
from app.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic - delete all collections
    collection_names = chroma_client.list_collections()
    logger.info(f"Found {len(collection_names)} collections")

    # Delete each collection
    for collection in collection_names:
        collection_name = collection.name
        logger.info(f"Deleting collection: {collection_name}")
        chroma_client.delete_collection(name=collection_name)
    
    yield
    # Shutdown logic (if needed) would go here


app = FastAPI(title="ChromaDB ORM Server", lifespan=lifespan)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)