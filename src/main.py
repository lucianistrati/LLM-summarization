from models.gpt2 import summarize_text_gpt2
from models.vertext_ai_model import vertext_ai_summary
from models.openai_model import openai_summary
from models.bart import summarize_text_with_bart
from query_database import query_database
import numpy as np


filename_to_idx = {'PMC8325057.txt': 0, 'PMC8198544.txt': 1,
                   'Emerging Frontiers in Drug Delivery.txt': 2,
                   's41392-022-00950-y.txt': 3,
                   'Efficacy and Safety of the mRNA-1273 SARS-CoV-2 Vaccine.txt': 4,
                   's41392-022-01007-w.txt': 5, 'nanomaterials-10-00364-v2.txt': 6,
                   's41591-022-02061-1.txt': 7, '82_2020_217.txt': 8,
                   'pharmaceutics-12-00102-v2.txt': 9,
                   'Safety and Efficacy of the BNT162b2 mRNA Covid-19 Vaccine.txt': 10,
                   'biomedicines-11-00308-v2.txt': 11,
                   'mRNA vaccines — a new era.txt': 12}


def main():
    print("Paper recommendation with Streamlit")

    # User input text
    input_text = input("Enter your paragraph, you can copy paste some from - "
                       "data/examples.md: ")

    texts = np.load("data/texts.npy", allow_pickle=True)

    # Display original text and summary
    print("Original Text:", input_text)
    result = query_database(input_text)

    scores = [value['score'] for value in result]
    texts = [texts[filename_to_idx[value["paper"]]] for value in result]

    for i in range(len(texts)):
        print("#" * 150)
        print(f"Retrieved paper: {result[i]['paper']}")
        print(f"Matching score: {scores[i]}")

        print(f"Paper no. {i + 1 } - {result[i]['paper']} Summary (Bart):")
        print(summarize_text_with_bart(texts[i]))

        print(f"Paper no. {i + 1} - {result[i]['paper']} Summary (GPT2):")
        print(summarize_text_gpt2(texts[i]))

        print(f"Paper no. {i + 1} - {result[i]['paper']} Summary (Open AI):")
        print(openai_summary(texts[i][:27000]))
        # there is an issue with the context
        # size limit for open ai, thus the 27k characters limit

        print(f"Paper no. {i + 1} - {result[i]['paper']} Summary (Vertex AI):")
        print(vertext_ai_summary(texts[i]))


if __name__ == "__main__":
    main()
