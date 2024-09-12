import fitz  # PyMuPDF
import pytesseract  # Tesseract for OCR
from PIL import Image  # For image processing
from io import BytesIO
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import camelot  # For table extraction

def extract_text_images_tables_from_page(page, doc, file_path, page_num):
    """Extract text, image-based text (OCR), and table data from a PDF page."""
    text = page.get_text("text")  # Extract vector text (standard text from PDF)
    image_text = ""

    # Extract tables from the page using Camelot
    try:
        tables = camelot.read_pdf(file_path, pages=str(page_num + 1))
        table_text = ""
        for table in tables:
            table_text += f"\n[TABLE DATA]\n{table.df.to_string()}\n"
    except Exception as e:
        table_text = f"\n[ERROR extracting tables on page {page_num + 1}]: {e}\n"

    # Extract images and apply OCR for text in images
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]  # Extract the image reference number
        base_image = fitz.Pixmap(doc, xref)  # Extract the image from the page
        
        # Convert to a PIL image for OCR processing
        img_data = Image.open(BytesIO(base_image.tobytes()))
        image_text += pytesseract.image_to_string(img_data)  # Perform OCR on the image

    # Combine vector text (if present), OCR results, and table data
    return text + "\n" + image_text + table_text

def process_pdf(file_path):
    # Open the PDF with PyMuPDF
    doc = fitz.open(file_path)
    pdf_text = ""
    
    # Iterate through each page in the document
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pdf_text += extract_text_images_tables_from_page(page, doc, file_path, page_num)
    
    # Split the text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create embeddings using Ollama Embeddings
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # Store the chunks and embeddings in Chroma
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory="ocr_rag_files")
    
    return docsearch

# Example usage: process the PDF and store the vector store
process_pdf("documents/Standard Operating Procedure (SOP) for stock verification and quality checking for sales of bulkpackaged FOMLFOMPROM under MDA scheme.pdf")
