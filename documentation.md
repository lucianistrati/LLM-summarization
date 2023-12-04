# Documentation

## Embeddings

- Sentence transformers - used
- Open AI Ada embeddings - tested
- Vertex AI embeddings - only this one is used in the final streamlit app

## Summaries extraction

- Bart
- GPT2
- Open AI (with GPT 3.5 / Chat GPT)
- Vertex AI (with Text Bison Text Generation model)

# Database
- Vector DB (with Pinecone), with 768 embedding dimension, we query the DB with an 
  embedding obtained through the same mechanism as the one used for populating the 
  DB and then use the cosine distance in order to retrieve the top k most similar ones.
 
# Deployment

- Streamlit app (where you have the option to enter a paragraph from a paper and to 
  then get recommended the top 3 most similar papers and 4 variants of summaries for 
  each of those 3 papers)

# Evaluation pipeline

The following metrics were used to compare the original texts with the summaries in 
order to then evaluate the summarization techniques:
- ROUGE Score - checks how much the words in the summary overlap with those in the 
  reference, higher scores mean the summary is closer to the reference;
- BLEU Score - measures how well the generated text matches a reference text, a 
  perfect match gets a score of 1, so also higher is better;
- Reading score - higher scores suggest easier readability and lower scores more 
  difficulty of understanding it;
- Fidelity score - looks at how well a summary preserves important details from the 
 original text with higher scores meaning better preservation of those details (in 
  this case named entities);
- Average of the 4 above.

## Results

################################################################################
Model used for summary extraction: GPT2
Rouge score: 0.3494033211880349
BLEU score: 0.01701453411634435
Reading score: 0.32048461538461537
Fidelity score: 0.018031284622900826
Average score: 0.17623343882797385
################################################################################
Model used for summary extraction: BART
Rouge score: 0.21879704772768174
BLEU score: 0.020507893541174684
Reading score: 0.15194615384615384
Fidelity score: 0.014412883537841384
Average score: 0.10141599466321291
################################################################################
Model used for summary extraction: Vertex AI - Text Bison Text Generation model
Rouge score: 0.22204717283173359
BLEU score: 0.014664775005946018
Reading score: 0.6032333333333334
Fidelity score: 0.012309652684829675
Average score: 0.21306373346396065
################################################################################
Model used for summary extraction: Open AI - Chat GPT (GPT 3.5)
Rouge score: 0.47569738576158077
BLEU score: 0.11626590578153928
Reading score: 0.2058846153846154
Fidelity score: 0.09441679247552548
Average score: 0.22306617485081526


## Issue with streamlit

There was one issue with streamlit, namely that when running it, it was not able to 
import vertexai, even though locally it worked, however the code where I tried to 
make it work is located commented here: 

```bash
 src/streamlit_app_breaking_commented.py
```
