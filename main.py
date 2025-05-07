from fastapi import FastAPI
from app.api.routes import router
from app.db.client import chroma_client


app = FastAPI(title="ChromaDB server")

app.include_router(router)

#Event to delete all the collections when the app starts
@app.on_event("startup")
async def on_startup():
    collection_names = chroma_client.list_collections()
    print(f"Found {len(collection_names)} collections")

    # Delete each collection
    for collection in collection_names:
        collection_name = collection.name
        print(f"Deleting collection: {collection_name}")
        chroma_client.delete_collection(name=collection_name)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)