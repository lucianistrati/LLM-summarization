from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm
import os
from statistics import mean
from textstat import flesch_reading_ease  # Requires 'textstat' library
import spacy

nlp = spacy.load("en_core_web_sm")

rouge = Rouge()


def calculate_flesch_reading_ease(text):
    return flesch_reading_ease(text)


def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]


def calculate_fidelity(original, summary):
    source_entities = set(extract_entities(original))
    generated_entities = set(extract_entities(summary))
    overlap = len(source_entities.intersection(generated_entities))
    fidelity_score = overlap / len(source_entities) if len(source_entities) > 0 else 0
    return fidelity_score


def evaluation_pipeline(summary, original):
    rouge_scores = rouge.get_scores(summary, original)
    rouge_score = mean([mean(list(values.values())) for values in
                        rouge_scores[0].values()])
    bleu_score = sentence_bleu([word_tokenize(summary)], word_tokenize(original))
    reading_score = calculate_flesch_reading_ease(summary) / 100.0
    fidelity_score = calculate_fidelity(original, summary)
    return (rouge_score, bleu_score,
            reading_score,
            fidelity_score)


def evaluate_summaries(summaries, original_texts, model_name):
    rouge_scores = []
    bleu_scores = []
    reading_scores = []
    fidelity_scores = []
    for (summary, original) in zip(summaries, original_texts):
        # import pdb
        # pdb.set_trace()
        if not len(str(summary)) or not len(original):
            continue
        (rouge_score, bleu_score,
         reading_score, fidelity_score) = evaluation_pipeline(str(summary), original)
        rouge_scores.append(rouge_score)
        bleu_scores.append(bleu_score)
        reading_scores.append(reading_score)
        fidelity_scores.append(fidelity_score)

    average_result =(mean(rouge_scores) + mean(bleu_scores) + mean(reading_scores) +
                     mean(fidelity_scores)) / 4
    print("#" * 80)
    print(f"Model used for summary extraction: {model_name}")
    print(f"Rouge score: {mean(rouge_scores)}")
    print(f"BLEU score: {mean(bleu_scores)}")
    print(f"Reading score: {mean(reading_scores)}")
    print(f"Fidelity score: {mean(fidelity_scores)}")
    print(f"Average score: {average_result}")


def main():
    bart_summaries = np.load(file="data/summaries/bart_summaries.npy",
                             allow_pickle=True)
    openai_summaries = np.load(file="data/summaries/open_ai_summaries.npy",
                             allow_pickle=True)
    vertex_ai_summaries = np.load(file="data/summaries/vertex_ai_summaries.npy",
                                allow_pickle=True)
    gpt2_summaries = np.load(file="data/summaries/gpt2_summaries.npy",
                             allow_pickle=True)
    folder_path = "extracted-text"
    filenames = os.listdir(folder_path)
    original_texts = []
    for i, filename in tqdm(enumerate(sorted(filenames))):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r") as f:
            text = f.read()
        original_texts.append(text)

    evaluate_summaries(bart_summaries, original_texts, "BART")
    evaluate_summaries(openai_summaries, original_texts, "Open AI")
    evaluate_summaries(vertex_ai_summaries, original_texts, "Vertex AI")
    evaluate_summaries(gpt2_summaries, original_texts, "GPT2")


if __name__ == "__main__":
    main()
