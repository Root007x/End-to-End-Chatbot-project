from flask import Flask, render_template, request, jsonify, session
from src.helper import load_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN



index_name = "constitution"

# load embedding
embedding = load_embedding_model()

# load vector database
vector_store_load = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

# Create retriever
retriever = vector_store_load.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" : 3}
)

# init model
llm_endpoint = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.3-70B-Instruct"
)
model = ChatHuggingFace(llm = llm_endpoint)

# prompt
prompt = ChatPromptTemplate([
    ('system' , system_prompt),
    ('placeholder', "{chat_history}"),
    ('human', "{input}")
])

# create chain
qa_chain = create_stuff_documents_chain(llm = model, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)



# app part
app = Flask(__name__)
app.secret_key = os.urandom(24)

user_memory = {} # "user_id" : memory_object

def get_user_memory(user_id):
    if user_id not in user_memory:
        user_memory[user_id] = ConversationBufferMemory(
                                    memory_key="chat_history",
                                    return_messages=True
                                )
    return user_memory[user_id]


@app.route("/")
def home():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/chat", methods = ["GET","POST"])
def chat():
    data = request.get_json()
    message = data.get('message')

    user_id = session["user_id"]
    memory = get_user_memory(user_id)

    chat_history = memory.load_memory_variables({})["chat_history"]
    input = {
        "input" : message,
        "chat_history" : chat_history
    }

    response = rag_chain.invoke(input)
    answer = response["answer"]
    memory.save_context({"input" : message},{"answer" : answer})

    return jsonify({'reply' : answer})


@app.route("/reset", methods = ["POST"])
def reset():
    user_id = session.get("user_id")

    if not user_id:
        return jsonify({"Status" : "No user id in session"})
    
    if user_id in user_memory:
        user_memory[user_id].clear()
    return jsonify({"Status" : "Memory Cleared"})


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 7860)