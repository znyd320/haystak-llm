from haystack import Document
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
import uuid
from weaviate import Client

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
document_store = WeaviateDocumentStore(host='http://localhost', port=8080, embedding_dim=384)  # Ensure embedding_dim matches the model

# Clear the document store by deleting the entire class
client = Client("http://localhost:8080")
client.schema.delete_all()
print("Cleared the document store.")

# MongoDB PDF converter
converter = MongoDBPDFConverter(MONGO_URI, DB_NAME, COLLECTION_NAME)
docs = converter.convert()

# Prepare documents for processing
final_doc = []
for doc in docs:
    new_doc = Document(
        content=doc['content'],
        meta=doc['meta'],
        id=str(uuid.uuid4())  # Generate a UUID for each document
    )
    final_doc.append(new_doc)

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,
    split_overlap=0  # You can adjust this based on your requirements
)

preprocessed_docs = preprocessor.process(final_doc)

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Calculate embeddings for preprocessed documents
embeddings = retriever.embed_documents(preprocessed_docs)

# Attach embeddings to the documents
for doc, embedding in zip(preprocessed_docs, embeddings):
    doc.embedding = embedding

# Write preprocessed documents with embeddings to the document store
document_store.write_documents(preprocessed_docs)

# Debugging: print document content types and first few docs
print("First few preprocessed documents:", preprocessed_docs[:1])
for doc in preprocessed_docs[:1]:
    print(f"Document ID: {doc.id}, Content Type: {type(doc.content)}, Meta: {doc.meta}, Embedding: {type(doc.embedding)}")

def update_embeddings_in_batches(document_store, retriever, batch_size=1000):
    try:
        client = document_store.weaviate_client
        client.batch.configure(batch_size=batch_size, dynamic=True)

        docs = document_store.get_all_documents(batch_size=batch_size, return_embedding=False)

        while docs:
            print(f"Processing batch of {len(docs)} documents")
            for doc in docs:
                # print(f"Doc ID: {doc.id}, Content Type: {type(doc.content)}, Content: {doc.content[:100]}")
                # Ensure no method upper() is called on content or Document
                if hasattr(doc, 'content') and isinstance(doc.content, str):
                    print(f"Before doc content: {doc.content[:100]}")
                    doc.content = doc.content.upper()
                    print(f"After doc content: {doc.content[:100]}")
                if hasattr(doc, 'meta') and isinstance(doc.meta, dict):
                    print(f"Before doc meta: {doc.meta}")
                    doc.meta = dict.fromkeys((k.upper() for k in doc.meta), doc.meta.values())
                    print(f"After doc meta: {doc.meta}")
            document_store.update_embeddings(retriever, docs)
            docs = document_store.get_all_documents(batch_size=batch_size, return_embedding=False)
    except Exception as e:
        print(f"Error during updating embeddings: {e}")

# Add try-except block here
try:
    # Update embeddings in batches
    print("Updating embeddings...")
    update_embeddings_in_batches(document_store, retriever)
except AttributeError as e:
    print(f"AttributeError: {e}")

print("Embeddings Done.")