
from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
import os
import fitz


def convert_pdf_to_string(file_path):
    output_string = StringIO()
    with open(file_path, 'rb') as in_file:
        parser = PDFParser(in_file) 
        doc = PDFDocument(parser) 
        rsrcmgr = PDFResourceManager() 
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
            
    return(output_string.getvalue())

def convert_to_txt(self, pdf_path: str) -> None:
    '''Converts a PDF file to a txt file using fitz (PyMuPDF).'''
    print("\nConverting to txt...")
    text = ""
    name = pdf_path.replace("pdf", "txt")
    text_file = open(name, "w", encoding="utf-8")
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        page = doc.load_page(i)
        text += page.get_text()
    text_file.write(text)
    text_file.close()

def split_to_title_and_pagenum(table_of_contents_entry):
    title_and_pagenum = table_of_contents_entry.strip()
    title = None
    pagenum = None
    if len(title_and_pagenum) > 0:
        if title_and_pagenum[-1].isdigit():
            i = -2
            while title_and_pagenum[i].isdigit():
                i -= 1
            title = title_and_pagenum[:i].strip()
            pagenum = int(title_and_pagenum[i:].strip())
    return title, pagenum


def get_publication_document_data(publication_path, publication_filename, data_type: str = None) -> tuple[fitz.Document, dict]:
    '''
    From a PDF file, get the doc object and publication metadata as dict to fit DB model .
    
    Args:
        publication_path: (str) - path to the PDF file.
        publication_filename: (str) -publication filename with .pdf extensnion
    Returns:
        tuple(fitz.Document, publication_metadata, publication_original_name) - doc object and publication metadata.
    '''
    document = fitz.open(publication_path)
    if not document.is_pdf:
        raise ValueError("File is not a PDF.")
    pdf_info = document.metadata
    keywords = pdf_info['keywords'] if 'keywords' in pdf_info else None
    filename = os.path.basename(publication_filename).replace(".pdf", "")
    publication_metadata = {
        "author": pdf_info['author'],
        "path": document.name,
        "filename": filename+".pdf",
        "type": data_type.value if hasattr(data_type, 'value') else data_type,
        "creation_date_raw": pdf_info['creationDate'],
        "title": pdf_info['title'] if pdf_info['title'] is not None else filename,
        "keywords": keywords
    }
    return document, publication_metadata
