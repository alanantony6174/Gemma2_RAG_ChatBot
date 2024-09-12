import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def process_pdf(file_path):
    # Read the PDF file
    pdf = PyPDF2.PdfReader(file_path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    # embeddings = OllamaEmbeddings(model="mxbai-embed-large", num_gpu=0)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # Note: You don't need to specify a path for persistence.
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory="rag_files")
    
    # Chroma automatically persists data; no need to manually call persist()
    return docsearch

# Example usage: process the PDF and store the vector store
process_pdf("documents/Alphadroid_HR_Handbooks.pdf")
