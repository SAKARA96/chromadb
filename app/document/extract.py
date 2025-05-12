import asyncio
import re
import os
from io import BytesIO
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
from app.logger import logger
import torch
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Determine device based on platform capabilities
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = 'mps'  # For Apple Silicon (macOS)
else:
    device = 'cpu'

# Load model with correct device
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

#text_splitter to chunk input document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      
    chunk_overlap=50,    
    separators=["\n\n", "\n", ".", " ", ""],
)

#---------------------------------------------------------------------------------------------------------------

def clean_text(text: str) -> str:
    # Remove escape sequences like \n, \r, \t, etc. (both actual and literal)
    text = re.sub(r'\\[nrtfbv]', ' ', text)              # Literal backslash escapes
    text = re.sub(r'[\n\r\t\f\v]', ' ', text)            # Actual control characters
    text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)        # Hex escape sequences
    text = re.sub(r'\s+', ' ', text)                     # Collapse multiple spaces
    return text.strip()

#---------------------------------------------------------------------------------------------------------------

async def extract_text(file_obj, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()

    logger.debug(f"Extracting text from file: {filename}, detected extension: {ext}")

    try:
        if ext == ".txt":
            logger.debug(f"Processing .txt file: {filename}")
            # For .txt files, directly read from the file object asynchronously
            content = await asyncio.to_thread(file_obj.read)
            text = content.decode("utf-8", errors="ignore")
            logger.debug(f"Successfully extracted text from .txt file: {filename}")

        elif ext == ".pdf":
            logger.debug(f"Processing .pdf file: {filename}")
            # Wrap pdfplumber in a thread for non-blocking operation
            text = await asyncio.to_thread(process_pdf, file_obj)
            logger.debug(f"Successfully extracted text from .pdf file: {filename}")

        elif ext == ".docx":
            logger.debug(f"Processing .docx file: {filename}")
            # Wrap docx processing in a thread for non-blocking operation
            text = await asyncio.to_thread(process_docx, file_obj)
            logger.debug(f"Successfully extracted text from .docx file: {filename}")

        else:
            logger.debug(f"Unsupported file type for file: {filename}. File extension: {ext}")
            raise ValueError("Unsupported file type. Use .txt, .pdf, or .docx.")

        return clean_text(text)

    except Exception as e:
        logger.debug(f"Error occurred while extracting text from file: {filename}. Error: {str(e)}", exc_info=True)
        raise  # Re-raise the exception to propagate it further

#---------------------------------------------------------------------------------------------------------------

# Helper function to handle PDF processing in a separate thread
def process_pdf(file_obj):
    with pdfplumber.open(file_obj) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

# Helper function to handle DOCX processing in a separate thread
def process_docx(file_obj):
    file_bytes = BytesIO(file_obj.read())
    doc = docx.Document(file_bytes)
    return "\n".join(p.text for p in doc.paragraphs)


# Modify the embedding generation function to use SentenceTransformer
async def generate_embeddings(texts: List[str]):
    return await asyncio.to_thread(_generate_embeddings, texts)

def _generate_embeddings(texts: List[str]) -> List[torch.Tensor]:
    embeddings_list: List[torch.Tensor] = []
    for text in texts:
        embedding = model.encode(text, convert_to_tensor=True)
        embeddings_list.append(embedding)
    return embeddings_list
