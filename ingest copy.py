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
import os
from dotenv import load_dotenv
import nltk
import warnings

# Filter out specific warnings
warnings.filterwarnings("ignore", message="Dep005: You are using weaviate-client version 3.26.4. The latest version is 4.6.5.")

load_dotenv()
nltk.download('punkt')  # Download the required NLTK data files

print("Import Successfully")


class MongoDBPDFConverter:
    def __init__(self, mongo_uri, db_name, collection_name):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def convert(self):
        docs = []
        for record in self.collection.find():
            content = record['pdf_data']
            meta = {"source_url": record['source_url'], "pdf_name": record['pdf_name']}
            docs.append({"content": content, "meta": meta})
        return docs


# MongoDB connection details
MONGO_URI = "mongodb://root:law_scrap_db@localhost:27018/"
DB_NAME = "law_scraper_db"
COLLECTION_NAME = "pdf_files"

# Document store setup
document_store = WeaviateDocumentStore(host='http://localhost', port=8080, embedding_dim=768)

# MongoDB PDF converter
converter = MongoDBPDFConverter(MONGO_URI, DB_NAME, COLLECTION_NAME)
docs = converter.convert()
print("#####################")

final_doc = []
for doc in docs:
    new_doc = {
        'content': doc['content'],
        'meta': doc['meta']
    }
    final_doc.append(new_doc)
    print("#####################")

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False,
    clean_header_footer=True,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,
)
print("#####################")

preprocessed_docs = preprocessor.process(final_doc)
print("#####################")

document_store.write_documents(preprocessed_docs)

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
print("Retriever: ", retriever)


def update_embeddings_in_batches(document_store, retriever, batch_size=1000):
    try:
        client = document_store.weaviate_client
        client.batch.configure(batch_size=batch_size, dynamic=True)

        docs = document_store.get_all_documents(batch_size=batch_size, return_embedding=False)
        while docs:
            document_store.update_embeddings(retriever, docs)
            docs = document_store.get_all_documents(batch_size=batch_size, return_embedding=False)
    except Exception as e:
        print(f"Error during updating embeddings: {e}")


update_embeddings_in_batches(document_store, retriever)
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
#     uvicorn.run("__main__:app", host='0.0.0.0', port=8001, reload=True)
