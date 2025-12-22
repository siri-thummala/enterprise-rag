import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma  # Updated from langchain_community.vectorstores

load_dotenv()

def start_ingestion(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # This creates the database folder
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("--- Success! Database created in ./chroma_db ---")

if __name__ == "__main__":
    start_ingestion("test.pdf")

