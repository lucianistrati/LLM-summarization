import vertexai

from vertexai.language_models import TextGenerationModel



vertexai.init(project="intense-wares-407020", location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.01,
    "top_p": 0.8,
    "top_k": 40
}

model = TextGenerationModel.from_pretrained("text-bison")


def vertext_ai_summary(text):
    response = model.predict(
        f"""Summarize this paper: {text}""",
        **parameters
    )
    return response.text
