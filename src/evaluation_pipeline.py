from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm
import os
from statistics import mean
from textstat import flesch_reading_ease
import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")
rouge = Rouge()

def calculate_flesch_reading_ease(text):
    """Calculate Flesch Reading Ease score for a given text."""
    return flesch_reading_ease(text)

def extract_entities(text):
    """Extract named entities from text using SpaCy."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def calculate_fidelity(original, summary):
    """Calculate the fidelity score based on entity overlap between original text and summary."""
    source_entities = set(extract_entities(original))
    generated_entities = set(extract_entities(summary))
    overlap = len(source_entities.intersection(generated_entities))
    return overlap / len(source_entities) if source_entities else 0

def evaluation_pipeline(summary, original):
    """Compute various evaluation metrics (ROUGE, BLEU, readability, fidelity) for a summary."""
    rouge_scores = rouge.get_scores(summary, original)
    rouge_score = mean([mean(scores.values()) for scores in rouge_scores[0].values()])
    bleu_score = sentence_bleu([word_tokenize(original)], word_tokenize(summary))
    reading_score = calculate_flesch_reading_ease(summary) / 100.0
    fidelity_score = calculate_fidelity(original, summary)
    return rouge_score, bleu_score, reading_score, fidelity_score

def evaluate_summaries(summaries, original_texts, model_name):
    """Evaluate summaries against original texts and print the evaluation metrics."""
    rouge_scores, bleu_scores, reading_scores, fidelity_scores = [], [], [], []

    for summary, original in zip(summaries, original_texts):
        if not summary or not original:
            continue
        rouge, bleu, reading, fidelity = evaluation_pipeline(str(summary), original)
        rouge_scores.append(rouge)
        bleu_scores.append(bleu)
        reading_scores.append(reading)
        fidelity_scores.append(fidelity)

    # Compute and display average scores
    average_score = mean([mean(rouge_scores), mean(bleu_scores), mean(reading_scores), mean(fidelity_scores)])
    print("#" * 80)
    print(f"Model used for summary extraction: {model_name}")
    print(f"ROUGE score: {mean(rouge_scores):.4f}")
    print(f"BLEU score: {mean(bleu_scores):.4f}")
    print(f"Reading score: {mean(reading_scores):.4f}")
    print(f"Fidelity score: {mean(fidelity_scores):.4f}")
    print(f"Average score: {average_score:.4f}")

def main():
    """Load summary files, read original texts, and evaluate summaries using different models."""
    # Load pre-saved summaries
    bart_summaries = np.load("data/summaries/bart_summaries.npy", allow_pickle=True)
    openai_summaries = np.load("data/summaries/open_ai_summaries.npy", allow_pickle=True)
    vertex_ai_summaries = np.load("data/summaries/vertex_ai_summaries.npy", allow_pickle=True)
    gpt2_summaries = np.load("data/summaries/gpt2_summaries.npy", allow_pickle=True)

    # Load original texts
    folder_path = "extracted-text"
    original_texts = []
    for filename in tqdm(sorted(os.listdir(folder_path))):
        with open(os.path.join(folder_path, filename), "r") as f:
            original_texts.append(f.read())

    # Evaluate summaries for each model
    evaluate_summaries(bart_summaries, original_texts, "BART")
    evaluate_summaries(openai_summaries, original_texts, "Open AI")
    evaluate_summaries(vertex_ai_summaries, original_texts, "Vertex AI")
    evaluate_summaries(gpt2_summaries, original_texts, "GPT-2")

if __name__ == "__main__":
    main()
