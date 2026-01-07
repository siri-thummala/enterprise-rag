def ingest_pdfs(pdf_folder):
    import os
    from dotenv import load_dotenv
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_chroma import Chroma  # Updated from langchain_community.vectorstores

    load_dotenv()


    all_documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = file
                all_documents.append(page)

    if not all_documents:
        raise ValueError("No PDF files found in the folder.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_documents)

    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # This creates the database folder
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("--- Success! Database created in ./chroma_db ---")

if __name__ == "__main__":
    start_ingestion("./pdfs")

