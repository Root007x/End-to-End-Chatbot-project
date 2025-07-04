from src.helper import load_pdf_file, load_embedding_model, text_split, init_vector_database
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# load .env
load_dotenv()

# load api keys
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


load_data = load_pdf_file(data = "data/")
text_chunks = text_split(extracted_data=load_data)
embedding = load_embedding_model()

# Pinecone vector database setup
index_name = "constitution"

init_vector_database(
    api_key=PINECONE_API_KEY,
    index_name= index_name,
    dimension=384
)

# store
vector_store = PineconeVectorStore.from_documents(
    embedding=embedding,
    index_name = index_name,
    documents = text_chunks
)



