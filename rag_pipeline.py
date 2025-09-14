from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define the path to your PDF file
pdf_path = "data/AI_and_Digital_Twin_Federation-Based_Flexible_Safety_Control_for_HumanRobot_Collaborative_Work_Cel.pdf"

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