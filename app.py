from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import WeaviateDocumentStore
from haystack import Pipeline

# Document store setup
document_store = WeaviateDocumentStore(host='http://localhost', port=8080, embedding_dim=384)  # Ensure embedding_dim matches the model

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

def get_result(query):
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

    response = pipeline.run(query=query, params={"Retriever": {"top_k": 5}})
    return response

# Question generated from "AUG§1Abs2Bek_07-1987.pdf"
results = get_result("Welche Staaten sind gemäß der Bekanntmachung über die Feststellung der Gegenseitigkeit gemäß § 1 Abs. 2 des Auslandsunterhaltsgesetzes vom 20. Juli 1987 (BGBl. 1987 II S. 420) als Gegenseitigkeitsstaaten anerkannt?")


# Question generated from "9._RAV.pdf"
# results = get_result("Welche Anpassungen der Rentenwerte und Unfallversicherungsleistungen in dem in Artikel 3 des Einigungsvertrages genannten Gebiet werden in der Verordnung zur neunten Anpassung der Renten (9. Rentenanpassungsverordnung - 9. RAV) vom 12. Dezember 1994 (BGBl. I S. 3805) festgelegt?")



for result in results["documents"]:
    print(f"\033[92mscore: {result.score}\033[0m")
    print(f"content: {result.content[0:100]}")
    print(f"metadata: {result.meta}")
    print("============================")

for key, value in results.items():
    if key != "documents":
        print(f"{key}: {value}")







# Model Entigration with HayStack done here;
# def get_result(query):
#     pipeline = Pipeline()
#     pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

#     prompt_model = PromptModel(model_name_or_path="gpt-3.5-turbo")
#     prompt_node = PromptNode(prompt_model)

#     prompt_template = PromptTemplate(
#         name="question-answering",
#         prompt_text="""Given the provided Documents, answer the Query. Make your answer detailed and long
#                        Query: {query}
#                        Documents: {join(documents)}
#                        Answer: """
#     )

#     pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])
#     pipeline.add_node(component=prompt_template, name="PromptTemplate", inputs=["PromptNode"])

#     response = pipeline.run(query=query, params={"Retriever": {"top_k": 5}})
#     return response