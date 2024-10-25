import os
import numpy as np
from tqdm import tqdm
from vectordb_api import VectorDBAPI, flatten
from text_embedder_processors import TextEmbedderProcessor, TextEmbeddingMethods
from models.openai_model import openai_summary
from models.bart import summarize_text_with_bart
from models.gpt2 import summarize_text_gpt2
from models.vertext_ai_model import vertext_ai_summary

def create_data():
    """Processes texts from a folder, generates summaries, embeddings, and saves them."""
    folder_path = "extracted-text"
    filenames = os.listdir(folder_path)
    
    # Initialize embedding processors and storage lists
    tp_st = TextEmbedderProcessor("", method=TextEmbeddingMethods.ST)
    tp_vertex = TextEmbedderProcessor("", method=TextEmbeddingMethods.VERTEX)
    bart_summaries, gpt2_summaries, vertex_ai_summaries, openai_summaries = [], [], [], []
    st_embs, vertex_embs = [], []

    # Process each file in the folder
    for filename in tqdm(sorted(filenames), desc="Generating summaries and embeddings"):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r") as f:
            text = f.read()

        # Generate and store summaries for each model
        openai_summaries.append(openai_summary(text[:min(len(text), 27_000)]))
        bart_summaries.append(summarize_text_with_bart(text))
        gpt2_summaries.append(summarize_text_gpt2(text))
        vertex_ai_summaries.append(vertext_ai_summary(text))

        # Generate and store embeddings for ST and Vertex models
        tp_st.text, tp_vertex.text = text, text
        st_embs.append(tp_st.process())
        vertex_embs.append(tp_vertex.process())

    # Save all summaries and embeddings
    np.save("data/summaries/bart_summaries.npy", np.array(bart_summaries), allow_pickle=True)
    np.save("data/summaries/open_ai_summaries.npy", np.array(openai_summaries), allow_pickle=True)
    np.save("data/summaries/vertex_ai_summaries.npy", np.array(vertex_ai_summaries), allow_pickle=True)
    np.save("data/summaries/gpt2_summaries.npy", np.array(gpt2_summaries), allow_pickle=True)
    np.save("data/embs/entire_st_embs.npy", np.array(st_embs), allow_pickle=True)
    np.save("data/embs/entire_vertex_embs.npy", np.array(vertex_embs), allow_pickle=True)

def load_data():
    """Loads embeddings and inserts text and embeddings into a vector database."""
    folder_path = "extracted-text"
    filenames = os.listdir(folder_path)
    vector_db_api = VectorDBAPI(index_name="main", renew_index=True)
    vertex_embs = np.load("data/embs/entire_vertex_embs.npy", allow_pickle=True)

    # Insert each text and its corresponding embedding into the vector database
    for i, filename in tqdm(enumerate(sorted(filenames)), desc="Inserting into VectorDB"):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r") as f:
            text = f.read()
        embedding = vertex_embs[i]
        final_content = f"{filename}---{text}"
        
        # Insert into the vector database with content truncated to 511 characters
        vector_db_api.insert(
            text=final_content[:min(511, len(final_content))],
            embedding=list(flatten(embedding))
        )

def main():
    """Main function to create data and load it into the database."""
    create_data()
    load_data()

if __name__ == "__main__":
    main()
