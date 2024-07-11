from haystack.nodes import EmbeddingRetriever, PreProcessor
from haystack.document_stores import WeaviateDocumentStore
from haystack import Document, Pipeline
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn
import uuid

# Document store setup
document_store = WeaviateDocumentStore(
    host='http://localhost', port=8080, embedding_dim=384)

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

app = FastAPI()

# Body payload type of ingest
# {
#  "content": "Welche Staaten sind gemäß der Bekanntmachung über die Feststellung der Gegenseitigkeit gemäß § 1 Abs. 2 des Auslandsunterhaltsgesetzes vom 20. Juli 1987 (BGBl. 1987 II S. 420) als Gegenseitigkeitsstaaten anerkannt?",
#  "meta": {
#    "source_url": "https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=20000516&FassungVom=20220101",
#    "pdf_name": "AUG§1Abs2Bek_07-1987.pdf"
# }


@app.post("/ingest")
async def ingest(request: Request):
    data = await request.json()
    content = data.get("content")
    meta = data.get("meta")

    if not content or not meta:
        return Response("Both 'content' and 'meta' are required.", status_code=400)

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=500,
        split_respect_sentence_boundary=True,
        split_overlap=0
    )

    doc = Document(content=content, meta=meta, id=str(uuid.uuid4()))
    preprocessed_docs = preprocessor.process([doc])

    pdf_name = meta.get("pdf_name")
    if pdf_name:
        existing_docs = document_store.get_all_documents(
            filters={"meta.pdf_name": pdf_name})
        if existing_docs:
            return Response("Document already exists in the document store.", status_code=400)
    else:
        # Handle the case where "pdf_name" is missing in the meta dictionary
        return Response("'pdf_name' is missing in the 'meta' dictionary.", status_code=400)

    embeddings = retriever.embed_documents(preprocessed_docs)

    # print(embeddings)
    for doc, emb in zip(preprocessed_docs, embeddings):
        doc.embedding = emb

    document_store.write_documents(preprocessed_docs)

    # preprocessed_docs[0].embedding = embeddings[0]

    # document_store.write_documents(preprocessed_docs)

    return Response("Document ingested successfully.", status_code=200)


def get_result(query):
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

    response = pipeline.run(query=query, params={"Retriever": {"top_k": 5}})
    return response


@app.post("/result")
async def result(request: Request):
    data = await request.json()
    query = data.get("query")

    if not query:
        return Response("'query' is required.", status_code=400)

    results = get_result(query)
    response_data = jsonable_encoder(results)
    return JSONResponse(content=response_data)


if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=3001)
