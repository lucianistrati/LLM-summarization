from transformers import GPT2Tokenizer, GPT2LMHeadModel


def summarize_text_gpt2(text, max_length=513, model_name="gpt2"):
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary using GPT-2
    summary_ids = model.generate(input_ids, max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
