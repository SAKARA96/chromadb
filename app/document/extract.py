import os
import re
from io import BytesIO
import pdfplumber
import docx

def clean_text(text: str) -> str:
    # Remove escape sequences like \n, \r, \t, etc. (both actual and literal)
    text = re.sub(r'\\[nrtfbv]', ' ', text)              # Literal backslash escapes
    text = re.sub(r'[\n\r\t\f\v]', ' ', text)            # Actual control characters
    text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)        # Hex escape sequences
    text = re.sub(r'\s+', ' ', text)                     # Collapse multiple spaces
    return text.strip()

def extract_text(file_obj, filename: str) -> str:
    """
    Extract and clean text from a given file object based on the file extension.

    Args:
        file_obj: The file object to extract text from
        filename: The name of the file being processed

    Returns:
        A cleaned string containing the extracted text.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        content = file_obj.read()
        text = content.decode("utf-8", errors="ignore")

    elif ext == ".pdf":
        file_obj.seek(0)
        with pdfplumber.open(file_obj) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    elif ext == ".docx":
        file_bytes = BytesIO(file_obj.read())
        doc = docx.Document(file_bytes)
        text = "\n".join(p.text for p in doc.paragraphs)

    else:
        raise ValueError("Unsupported file type. Use .txt, .pdf, or .docx.")

    return clean_text(text)
