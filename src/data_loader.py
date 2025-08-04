import os
import pdfplumber

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file using pdfplumber, page by page.
    Returns the combined text from all pages.
    """
    with pdfplumber.open(file_path) as pdf:
        texts = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n".join(texts)

def extract_text(file_path):
    """
    Wrapper function for text extraction. 
    Supports only PDF files. Raises an error for unsupported formats.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext != '.pdf':
        raise ValueError(f"❌ Unsupported file type: {ext}. Only PDF files are allowed.")
    return extract_text_from_pdf(file_path)
