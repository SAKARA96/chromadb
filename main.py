from fastapi import FastAPI
from app.api.routes import router
from app.db.client import chroma_client

app = FastAPI(title="ChromaDB server")

app.include_router(router)

# Event to create the collection when the app starts
@app.on_event("startup")
async def on_startup():
    # Ensure that the 'test' collection is created
    if "test" not in chroma_client.list_collections():
        chroma_client.create_collection("test")
        print("Collection 'test' created.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
