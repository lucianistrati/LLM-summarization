from vectordb_api import VectorDBAPI
from text_embedder_processors import TextEmbedderProcessor


def query_database(text: str):
    vector_db_api = VectorDBAPI()
    embedding = TextEmbedderProcessor(text).process()
    output = vector_db_api.query(embedding=embedding, top_k=3)
    matches = output["matches"]
    result = [{"paper": match["id"][:match["id"].find("---")], "score": match[
        "score"]} for match in matches]
    return result
