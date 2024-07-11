from haystack.nodes import EmbeddingRetriever, PreProcessor, PromptModel, PromptNode, PromptTemplate
from haystack.document_stores import WeaviateDocumentStore
from haystack import Pipeline
from pymongo import MongoClient
from fastapi import FastAPI, Request, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
import uvicorn
import json
import re
import os
from dotenv import load_dotenv
import nltk
import warnings
from concurrent.futures import ThreadPoolExecutor
from pymongo.errors import CursorNotFound

# Filter out specific warnings
warnings.filterwarnings("ignore", message="Dep005: You are using weaviate-client version 3.26.4. The latest version is 4.6.5.")

load_dotenv()
nltk.download('punkt')  # Download the required NLTK data files

print("Import Successfully")

class MongoDBDataRetriever:
    def __init__(self, mongo_uri, db_name, collection_name, batch_size=1000):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.batch_size = batch_size

    def retrieve_data(self):
        cursor = self.collection.find()
        while True:
            try:
                batch = cursor.next()
                yield batch
            except CursorNotFound:
                break

# MongoDB connection details
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Document store setup
document_store = WeaviateDocumentStore(host='http://localhost', port=8080, embedding_dim=768)

# MongoDB data retriever
data_retriever = MongoDBDataRetriever(MONGO_URI, DB_NAME, COLLECTION_NAME, batch_size=1000)

def process_batch(batch):
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=False,
        clean_header_footer=True,
        split_by="word",
        split_length=500,
        split_respect_sentence_boundary=True,
    )
    preprocessed_docs = preprocessor.process(batch)
    document_store.write_documents(preprocessed_docs)
    return preprocessed_docs

print("Starting data retrieval and processing...")

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for batch in data_retriever.retrieve_data():
        futures.append(executor.submit(process_batch, batch))

    for future in futures:
        result = future.result()

print("Data processing complete.")

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

print("Retriever: ", retriever)

document_store.update_embeddings(retriever)

print("Embeddings Done.")

app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_result(query):
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

    prompt_model = PromptModel(model_name_or_path="gpt-3.5-turbo")
    prompt_node = PromptNode(prompt_model)

    prompt_template = PromptTemplate(
        name="question-answering",
        prompt_text="""Given the provided Documents, answer the Query. Make your answer detailed and long
                       Query: {query}
                       Documents: {join(documents)}
                       Answer: """
    )

    pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])
    pipeline.add_node(component=prompt_template, name="PromptTemplate", inputs=["PromptNode"])

    response = pipeline.run(query=query, params={"Retriever": {"top_k": 5}})
    return response

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_answer")
async def get_answer(request: Request, question: str = Form(...)):
    response = get_result(question)
    response_data = jsonable_encoder(response)
    return Response(response_data)

# Uncomment below to run the app directly
# if __name__ == "__main__":
#     uvicorn.run("app:app", host='0.0.0.0', port=8001, reload=True)
