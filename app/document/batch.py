
from fastapi import UploadFile
from app.document.extract import extract_text,generate_embeddings
from app.db.client import add_to_collection,top_1_collection,get_or_create_collection
from app.logger import logger

#---------------------------------------------------------------------------------------------------------------

async def process_file(file: UploadFile, file_map: dict):
    try:
        logger.info(f"Started processing file: {file.filename}")
        text = await extract_text(file.file, file.filename)
        
        file_map[file.filename]["text"] = {
            "content": text,                  
            "total_characters": len(text),    
            "error": None,                    
        }
        
        logger.info(f"Successfully extracted text from file: {file.filename}, with {len(text)} characters.")
        
    except ValueError as ve:
        logger.warning(f"ValueError while processing file {file.filename}: {str(ve)}")

        file_map[file.filename]["text"] = {
            "error": str(ve),
        }
        
    except Exception as e:
        logger.error(f"Error while processing file {file.filename}: {str(e)}", exc_info=True)

        file_map[file.filename]["text"] = {
            "error": f"Failed to process file: {str(e)}"
        }

    logger.info(f"Finished processing file: {file.filename}")

#---------------------------------------------------------------------------------------------------------------

async def process_text(filename: str, file_map: dict):
    try:
        logger.info(f"Started processing text for file: {filename}")

        if file_map[filename]["text"]["error"] is None:
            text = file_map[filename]["text"]["content"]
            logger.info(f"Generating embeddings for file: {filename}")

            embeddings = await generate_embeddings(text)

            file_map[filename]["embedding"] = {
                "content": embeddings.tolist(),
                "total_embeddings": len(embeddings),
                "error": None,
            }

            logger.info(f"Successfully generated embeddings for file: {filename} with {len(embeddings)} embeddings.")
        else:
            file_map[filename]["embedding"] = {
                "error": f"Failed to extract text",
            }
            logger.warning(f"Text extraction failed for file: {filename}. Embedding generation skipped.")

    except ValueError as ve:
        file_map[filename]["embedding"] = {
            "error": str(ve)
        }
        logger.warning(f"ValueError occurred while processing file {filename}: {str(ve)}")

    except Exception as e:
        file_map[filename]["embedding"] = {
            "error": f"Failed to generate embeddings file: {str(e)}"
        }
        logger.error(f"Error occurred while generating embeddings for file {filename}: {str(e)}", exc_info=True)

#---------------------------------------------------------------------------------------------------------------

async def process_embeddings(filename: str, file_map: dict):
    try:
        logger.info(f"Started processing embeddings for file: {filename}")

        if file_map[filename]["embedding"]["error"] is None:
            embedding = file_map[filename]["embedding"]["content"]
            text = file_map[filename]["text"]["content"]
            uuid = file_map[filename]["uuid"]

            logger.info(f"Embeddings are available for file: {filename}. Identifying closest collection.")
            
            collection_name = await top_1_collection(embedding)
            collection = get_or_create_collection(collection_name=collection_name)
            
            logger.info(f"Collection identified: {collection.name}. Adding embeddings to the collection.")
            
            await add_to_collection(text=text, embedding=embedding, doc_id=uuid, filename=filename, collection=collection)
            
            file_map[filename]["status"] = "success"
            file_map[filename]["collection_name"] = collection_name

            logger.info(f"Successfully added embeddings to collection {collection.name} for file: {filename}")

        else:
            file_map[filename]["status"] = "failed"
            file_map[filename]["error"] = {
                "error": "Embedding extraction failed"
            }
            logger.warning(f"Embedding extraction failed for file: {filename}. Skipping collection addition.")

    except ValueError as ve:
        file_map[filename]["status"] = "failed"
        file_map[filename]["error"] = {
            "error": str(ve)
        }
        logger.warning(f"ValueError while processing file {filename}: {str(ve)}")

    except Exception as e:
        file_map[filename]["status"] = "failed"
        file_map[filename]["error"] = {
            "error": f"Failed to process embeddings for file: {str(e)}"
        }
        logger.error(f"Error occurred while processing embeddings for file {filename}: {str(e)}", exc_info=True)

#---------------------------------------------------------------------------------------------------------------