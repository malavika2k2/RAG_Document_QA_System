import os
from dotenv import load_dotenv
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


### 1. Configuration & Setup ###

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from your .env file
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in your .env file.")

# Set the environment variable for langchain
os.environ["OPENAI_API_KEY"] = api_key

# Define the path to your PDF file
pdf_path = "data/AI_and_Digital_Twin_Federation-Based_Flexible_Safety_Control_for_HumanRobot_Collaborative_Work_Cel.pdf"


### 2. Document Loading & Splitting ###

print("Loading and splitting the document...")

# Load the document
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs_chunks = text_splitter.split_documents(documents)

print(f"Number of chunks created: {len(docs_chunks)}")


### 3. Embedding & Vector Store ###

# Create the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create the vector store from the document chunks
vector_store = Chroma.from_documents(
    docs_chunks, 
    embedding_model, 
    persist_directory="chroma_db"
)

print("Vector store created and saved successfully!")


### 4. Retriever & LLM Setup ###

# Create the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize the LLM with the correct OpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

# Create the RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


### 5. Final Application Loop ###

print("\nWelcome to the RAG Document QA System!")
print("Enter a question about the document, or type 'exit' to quit.")

while True:
    query = input("\nYour Question: ")
    if query.lower() == 'exit':
        print("Goodbye!")
        break
    
    result = qa_chain.invoke({"query": query})
    
    print("\n**Answer:**")
    print(result["result"])
    
    print("\n**Sources:**")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")