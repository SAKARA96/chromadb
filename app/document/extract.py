import asyncio
import re
import os
from io import BytesIO
import aiofiles
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text: str) -> str:
    # Remove escape sequences like \n, \r, \t, etc. (both actual and literal)
    text = re.sub(r'\\[nrtfbv]', ' ', text)              # Literal backslash escapes
    text = re.sub(r'[\n\r\t\f\v]', ' ', text)            # Actual control characters
    text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)        # Hex escape sequences
    text = re.sub(r'\s+', ' ', text)                     # Collapse multiple spaces
    return text.strip()

async def extract_text(file_obj, filename: str) -> str:
    """
    Extract and clean text from a given file object based on the file extension.
    Asynchronous version of extract_text that allows parallel processing.

    Args:
        file_obj: The file object to extract text from
        filename: The name of the file being processed

    Returns:
        A cleaned string containing the extracted text.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        # For .txt files, directly read from the file object asynchronously
        content = await asyncio.to_thread(file_obj.read)
        text = content.decode("utf-8", errors="ignore")


    elif ext == ".pdf":
        # Wrap pdfplumber in a thread for non-blocking operation
        text = await asyncio.to_thread(process_pdf, file_obj)

    elif ext == ".docx":
        # Wrap docx processing in a thread for non-blocking operation
        text = await asyncio.to_thread(process_docx, file_obj)

    else:
        raise ValueError("Unsupported file type. Use .txt, .pdf, or .docx.")

    return clean_text(text)

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
async def generate_embeddings(text: str):
    return await asyncio.to_thread(_generate_embeddings, text)

def _generate_embeddings(text: str):
    # Use SentenceTransformer to encode the text into embeddings
    embeddings = model.encode(text)
    return embeddings
