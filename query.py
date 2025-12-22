import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Update this specific line for 2025 compatibility:
from langchain_classic.chains import RetrievalQA 

load_dotenv()

def ask_pdf(question):
    # Use the same 004 model you used for ingestion
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # Use 2.5 Flash Lite
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever()
    )

    print(f"\n--- Searching PDF for: {question} ---")
    response = qa_chain.invoke({"query": question})
    print("\nAI Answer:", response["result"])

if __name__ == "__main__":
    query = input("Ask a question about your PDF: ")
    ask_pdf(query)
