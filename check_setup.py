import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# We are using 'gemini-2.0-flash' which is the 2025 standard
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

try:
    response = llm.invoke("Hello! I upgraded my Python and fixed the model name. Can you hear me?")
    print("--- Success! Gemini says: ---")
    print(response.content)
except Exception as e:
    print(f"Error: {e}")

