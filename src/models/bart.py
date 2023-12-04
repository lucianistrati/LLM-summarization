from transformers import BartTokenizer, BartForConditionalGeneration


def summarize_text_with_bart(text, max_length=1024,
                             model_name="facebook/bart-large-cnn"):
    # Load pre-trained BART model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024,
                                 truncation=True)

    # Generate summary using BART
    summary_ids = model.generate(input_ids, max_length=max_length, length_penalty=2.0,
                                 num_beams=4, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
