
from fastapi import UploadFile
from app.document.extract import extract_text,generate_embeddings
from app.db.client import add_to_collection,top_1_collection,get_or_create_collection
from app.api.request import Status,StatusEnum
from app.logger import logger
from torch import Tensor
from app.api.request import BaseDocument,UploadDocument
from app.document.extract import text_splitter

#---------------------------------------------------------------------------------------------------------------

async def process_file(file: UploadFile, upload_document: BaseDocument):
    try:
        logger.info(f"Started processing file: {file.filename}")
        text = await extract_text(file.file, file.filename)
        chunks = text_splitter.split_text(text)
        upload_document.text.content = chunks
        upload_document.text.error = None 
        upload_document.text.shape = [len(chunks)]
        
        logger.info(f"Successfully extracted text from file: {file.filename}, with {len(text)} characters.")
        
    except ValueError as ve:
        logger.warning(f"ValueError while processing file {file.filename}: {str(ve)}")
        upload_document.text.error = str(ve)
        
    except Exception as e:
        logger.error(f"Error while processing file {file.filename}: {str(e)}", exc_info=True)
        upload_document.text.error = f"Failed to process file: {str(e)}"

    logger.info(f"Finished processing file: {file.filename}")

#---------------------------------------------------------------------------------------------------------------

async def process_text(filename:str, upload_document: BaseDocument):
    try:
        logger.info(f"Started processing text for file: {filename}")

        if upload_document.text.error is None:
            text = upload_document.text.content
            logger.info(f"Generating embeddings for file: {filename}")

            embeddings = await generate_embeddings(text)
            upload_document.embedding.content = embeddings
            upload_document.embedding.shape = len(embeddings)
            upload_document.embedding.error = None
            upload_document.embedding.convert_to_list_floats()

            logger.info(f"Successfully generated embeddings for file: {filename} with {len(embeddings)} embeddings.")
        else:
            upload_document.embedding.error = f"Failed to extract text from process_file"
            logger.warning(f"Text extraction failed for file: {filename}. Embedding generation skipped.")

    except ValueError as ve:
        upload_document.embedding.error = str(ve)
        logger.warning(f"ValueError occurred while processing file {filename}: {str(ve)}")

    except Exception as e:
        upload_document.embedding.error = f"Failed to generate embeddings file: {str(e)}"
        logger.error(f"Error occurred while generating embeddings for file {filename}: {str(e)}", exc_info=True)

#---------------------------------------------------------------------------------------------------------------

async def process_embeddings(filename:str, upload_document: UploadDocument):
    try:
        logger.info(f"Started processing embeddings for file: {filename}")
        
        if upload_document.embedding.error is None:
            embedding = upload_document.embedding.content
            text = upload_document.text.content
            uuid = upload_document.uuid
            vectordb_embedding = upload_document.embedding.vectordb_embeddings

            logger.info(f"Embeddings are available for file: {filename}. Identifying closest collection.")
            
            collection_name = await top_1_collection(embedding)
            collection = get_or_create_collection(collection_name=collection_name)
            
            logger.info(f"Collection identified: {collection.name}. Adding embeddings to the collection.")
            
            await add_to_collection(text=text, embedding=vectordb_embedding, start_idx=0, filename=filename, collection=collection)
            
            upload_document.status = Status(code=StatusEnum.SUCCESS,error=None)
            upload_document.collection = collection_name

            logger.info(f"Successfully added embeddings to collection {collection.name} for file: {filename}")

        else:
            upload_document.status = Status(code=StatusEnum.FAILED,error="Embedding extraction failed")
            logger.warning(f"Embedding extraction failed for file: {filename}. Skipping collection addition.")

    except ValueError as ve:
        upload_document.status = Status(code=StatusEnum.FAILED,error= str(ve))
        logger.warning(f"ValueError while processing file {filename}: {str(ve)}")

    except Exception as e:
        upload_document.status = Status(code=StatusEnum.FAILED,error= f"Failed to process embeddings for file: {str(e)}")
        logger.error(f"Error occurred while processing embeddings for file {filename}: {str(e)}", exc_info=True)

#---------------------------------------------------------------------------------------------------------------