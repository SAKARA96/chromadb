
from fastapi import  UploadFile
from app.document.extract import extract_text,generate_embeddings
from app.db.client import add_to_collection


#Async function to process_file and extract text
async def process_file(file: UploadFile, file_map: dict):
    try:
        # Extract the text using the async extract_text function
        text = await extract_text(file.file, file.filename)
        file_map[file.filename]["text"] = {
            "content": text,
            "total_characters" : len(text),
            "error": None,
        }
    except ValueError as ve:
        file_map[file.filename]["text"] = {
            "error": str(ve),
        }
    except Exception as e:
        file_map[file.filename]["text"] = {
            "error": f"Failed to process file: {str(e)}"
        }

async def process_text(filename: str, file_map : dict):
    try:
        if file_map[filename]["text"]["error"] is None:
            text = file_map[filename]["text"]["content"]
            embeddings = await generate_embeddings(text)

            file_map[filename]["embedding"] = {
                "content": embeddings.tolist(),
                "total_embeddings" : len(embeddings),
                "error": None,
            }
        else:
            file_map[filename]["embedding"] = {
                "error": f"Failed to extract text",
            }
    except ValueError as ve:
        file_map[filename]["embedding"] = {
            "error": str(ve)
        }
    except Exception as e:
        file_map[filename]["embedding"] = {
            "error": f"Failed to generate embeddings file: {str(e)}"
        }

async def  process_embeddings(filename:str, file_map : dict):
    try:
        if file_map[filename]["embedding"]["error"] is None:
            embedding = file_map[filename]["embedding"]["content"]
            text = file_map[filename]["text"]["content"]
            uuid = file_map[filename]["uuid"]

            #Add to chroma db collection
            await add_to_collection(text=text, embedding=embedding, doc_id=uuid)
            file_map[filename]["status"] = "success"

    except ValueError as ve:
        file_map[filename]["status"] = "failed"
        file_map[filename]["error"] = {
            "error": str(ve)
        }
    except Exception as e:
        file_map[filename]["status"] = "failed"
        file_map[filename]["error"] = {
            "error": f"Failed to process embeddings file: {str(e)}"
        }