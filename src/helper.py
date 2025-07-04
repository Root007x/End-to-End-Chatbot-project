from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone.grpc import PineconeGRPC as pinecone
from pinecone import ServerlessSpec


# Load the PDFs file
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob = "*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

# Text split
def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 330,
        chunk_overlap = 20
    )
    chunk = splitter.split_documents(extracted_data)
    return chunk

# Load embedding Model
def load_embedding_model():
    embedding = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding

# Init database
def init_vector_database(api_key, index_name, dimension):
    pc = pinecone(api_key=api_key)

    index_name = index_name

    pc.create_index(
        name=index_name,
        dimension = dimension,
        metric = "cosine",
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )