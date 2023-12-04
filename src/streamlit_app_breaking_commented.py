# import streamlit as st
#
# # from models.langchain_model import langchain_summary
# from models.gpt2 import summarize_text_gpt2
# from models.vertext_ai_model import vertext_ai_summary
# from models.openai_model import openai_summary
# from models.bart import summarize_text_with_bart
# from query_database import query_database
# import numpy as np
#
#
# filename_to_idx = {'PMC8325057.txt': 0, 'PMC8198544.txt': 1,
#                    'Emerging Frontiers in Drug Delivery.txt': 2,
#                    's41392-022-00950-y.txt': 3,
#                    'Efficacy and Safety of the mRNA-1273 SARS-CoV-2 Vaccine.txt': 4,
#                    's41392-022-01007-w.txt': 5, 'nanomaterials-10-00364-v2.txt': 6,
#                    's41591-022-02061-1.txt': 7, '82_2020_217.txt': 8,
#                    'pharmaceutics-12-00102-v2.txt': 9,
#                    'Safety and Efficacy of the BNT162b2 mRNA Covid-19 Vaccine.txt': 10,
#                    'biomedicines-11-00308-v2.txt': 11,
#                    'mRNA vaccines — a new era.txt': 12}
#
#
# def main():
#     st.title("Paper recommendation with Streamlit")
#
#     # User input text
#     input_text = st.text_area("Enter your paragraph")
#
#     texts = np.load("data/texts.npy", allow_pickle=True)
#
#     if st.button("Summarize"):
#         if input_text:
#             # Display original text and summary
#             st.subheader("Original Text:")
#             st.write(input_text)
#
#             result = query_database(input_text)
#
#             scores = [value['score'] for value in result]
#             texts = [texts[filename_to_idx[value["paper"]]] for value in result]
#
#             for i in range(len(texts)):
#                 st.subheader("#" * 50)
#                 st.subheader(f"Matching score: {scores[i]}")
#
#                 st.subheader(f"Paper no. {i + 1} Summary (Bart):")
#                 st.write(summarize_text_with_bart(texts[i]))
#
#                 st.subheader(f"Paper no. {i + 1} (GPT2):")
#                 st.write(summarize_text_gpt2(texts[i]))
#
#                 st.subheader(f"Paper no. {i + 1} Summary (Open AI):")
#                 st.write(openai_summary(texts[i]))
#
#                 st.subheader(f"Paper no. {i + 1} Summary (Vertex AI):")
#                 st.write(vertext_ai_summary(texts[i]))
#         else:
#             st.warning("Please enter text to summarize.")
#
#
# if __name__ == "__main__":
#     main()
