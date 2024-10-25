import streamlit as st
import numpy as np
# Import summarization functions and database query
from models.gpt2 import summarize_text_gpt2
from models.vertext_ai_model import vertext_ai_summary
from models.openai_model import openai_summary
from models.bart import summarize_text_with_bart
from query_database import query_database

# Mapping of filenames to indices in the data array
filename_to_idx = {
    'PMC8325057.txt': 0, 'PMC8198544.txt': 1, 'Emerging Frontiers in Drug Delivery.txt': 2,
    's41392-022-00950-y.txt': 3, 'Efficacy and Safety of the mRNA-1273 SARS-CoV-2 Vaccine.txt': 4,
    's41392-022-01007-w.txt': 5, 'nanomaterials-10-00364-v2.txt': 6,
    's41591-022-02061-1.txt': 7, '82_2020_217.txt': 8, 'pharmaceutics-12-00102-v2.txt': 9,
    'Safety and Efficacy of the BNT162b2 mRNA Covid-19 Vaccine.txt': 10,
    'biomedicines-11-00308-v2.txt': 11, 'mRNA vaccines — a new era.txt': 12
}

def main():
    """Streamlit application for recommending and summarizing research papers."""
    st.title("Scientific Paper Recommendation and Summarization")

    # User input text
    input_text = st.text_area("Enter your paragraph")

    # Load text data
    texts = np.load("data/texts.npy", allow_pickle=True)

    if st.button("Summarize"):
        if input_text:
            st.subheader("Original Text:")
            st.write(input_text)

            # Query the database for similar papers
            result = query_database(input_text)
            scores = [value['score'] for value in result]
            matched_texts = [texts[filename_to_idx[value["paper"]]] for value in result]

            # Display summaries from each model for each matched text
            for i, text in enumerate(matched_texts):
                paper_name = result[i]["paper"]
                score = scores[i]

                st.markdown(f"---\n### Paper {i + 1}: {paper_name} (Score: {score})")

                # Display summaries by model
                st.write(f"**Bart Summary**: {summarize_text_with_bart(text)}")
                st.write(f"**GPT2 Summary**: {summarize_text_gpt2(text)}")
                st.write(f"**Open AI Summary**: {openai_summary(text[:27000])}")  # Handle OpenAI's context limit
                st.write(f"**Vertex AI Summary**: {vertext_ai_summary(text)}")
        else:
            st.warning("Please enter text to summarize.")

if __name__ == "__main__":
    main()
