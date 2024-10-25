from vectordb_api import VectorDBAPI
from text_embedder_processors import TextEmbedderProcessor

def query_database(text: str):
    """
    Queries the vector database for papers most relevant to the input text.

    Args:
        text (str): The input text to find similar papers for.

    Returns:
        list of dict: A list of dictionaries containing paper identifiers and 
                      their matching scores, sorted by relevance.
    """
    # Initialize database and embed the input text
    vector_db_api = VectorDBAPI()
    embedding = TextEmbedderProcessor(text).process()

    # Query the vector database for top 3 similar entries
    output = vector_db_api.query(embedding=embedding, top_k=3)
    matches = output["matches"]

    # Format result with paper names and similarity scores
    result = [{"paper": match["id"].split("---")[0], "score": match["score"]} for match in matches]
    
    return result
