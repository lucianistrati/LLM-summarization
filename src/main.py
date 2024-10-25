# Import model-specific summarization functions and utilities
from models.gpt2 import summarize_text_gpt2
from models.vertext_ai_model import vertext_ai_summary
from models.openai_model import openai_summary
from models.bart import summarize_text_with_bart
from query_database import query_database
import numpy as np

# Mapping from filename to index in the data array
filename_to_idx = {
    'PMC8325057.txt': 0, 'PMC8198544.txt': 1, 'Emerging Frontiers in Drug Delivery.txt': 2,
    's41392-022-00950-y.txt': 3, 'Efficacy and Safety of the mRNA-1273 SARS-CoV-2 Vaccine.txt': 4,
    's41392-022-01007-w.txt': 5, 'nanomaterials-10-00364-v2.txt': 6,
    's41591-022-02061-1.txt': 7, '82_2020_217.txt': 8, 'pharmaceutics-12-00102-v2.txt': 9,
    'Safety and Efficacy of the BNT162b2 mRNA Covid-19 Vaccine.txt': 10,
    'biomedicines-11-00308-v2.txt': 11, 'mRNA vaccines — a new era.txt': 12
}

def main():
    """Runs paper recommendation and summarization pipeline."""
    print("Paper recommendation with Streamlit")

    # User input prompt
    input_text = input("Enter your paragraph (you can copy-paste from data/examples.md): ")

    # Load texts from file
    texts = np.load("data/texts.npy", allow_pickle=True)

    # Display the input text
    print("Original Text:", input_text)

    # Retrieve and score relevant papers
    result = query_database(input_text)
    scores = [value['score'] for value in result]
    retrieved_texts = [texts[filename_to_idx[value["paper"]]] for value in result]

    # Iterate through retrieved papers, displaying summaries from each model
    for i, text in enumerate(retrieved_texts):
        paper_name = result[i]['paper']
        score = scores[i]
        
        print("#" * 150)
        print(f"Retrieved Paper: {paper_name}")
        print(f"Matching Score: {score}")

        # Display summaries from each model
        print(f"Paper {i + 1} - {paper_name} Summary (Bart):")
        print(summarize_text_with_bart(text))

        print(f"Paper {i + 1} - {paper_name} Summary (GPT2):")
        print(summarize_text_gpt2(text))

        print(f"Paper {i + 1} - {paper_name} Summary (Open AI):")
        print(openai_summary(text[:27000]))  # Limit due to OpenAI context size

        print(f"Paper {i + 1} - {paper_name} Summary (Vertex AI):")
        print(vertext_ai_summary(text))

if __name__ == "__main__":
    main()
