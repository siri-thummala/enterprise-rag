import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="📄",
    layout="centered"
)

st.title("📚 Enterprise RAG System")
st.write("Ask questions over your PDFs using Retrieval-Augmented Generation.")

# Initialize embeddings (same as ingestion)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

# Load vector database
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Load LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# User input
query = st.text_input("Ask a question about your PDFs:")

if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            # Retrieve relevant documents
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)


            # Build context
            context = "\n\n".join(doc.page_content for doc in docs)

            # Prompt (hallucination control)
            prompt = f"""
You are an assistant answering questions strictly using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""

            response = llm.invoke(prompt)

        # Display answer
        st.subheader("Answer")
        st.write(response.content)

        # Display sources
        st.subheader("Sources")
        sources = set()
        for doc in docs:
            src = doc.metadata.get("source")
            page = doc.metadata.get("page")
            sources.add(f"{src}, page {page}")

        for s in sources:
            st.write(f"- {s}")
