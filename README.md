# LLM summarization

## Create embeddings and insert them in the Vector DB

```bash
python src/populate_db.py
```

- create_data() function - for creation of the embeddings and of the summaries
- load_data() function - for insertion of the Vertex AI Text bison embeddings in the DB


## Run the tasks

```bash
python src/main.py
```

## Install dependencies

To download and install Poetry run:

```bash
make poetry-download
```

To uninstall

```bash
make poetry-remove
```

To install all the dependencies:

```bash
make install
```


## Running the streamlit app

```bash
 streamlit run src/app.py
```

## Documentation

### Embeddings

- Sentence transformers - used
- Open AI Ada embeddings - tested
- Vertex AI embeddings - only this one is used in the final streamlit app

### Summaries extraction

- Bart
- GPT2
- Open AI (with GPT 3.5 / Chat GPT)
- Vertex AI (with Text Bison Text Generation model)

### Database
- Vector DB (with Pinecone), with 768 embedding dimension, we query the DB with an 
  embedding obtained through the same mechanism as the one used for populating the 
  DB and then use the cosine distance in order to retrieve the top k most similar ones.
 
### Deployment

- Streamlit app (where you have the option to enter a paragraph from a paper and to 
  then get recommended the top 3 most similar papers and 4 variants of summaries for 
  each of those 3 papers)

### Evaluation pipeline

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

### Results

- Model used for summary extraction: GPT2
  - Rouge score: 0.35
  - BLEU score: 0.02
  - Reading score: 0.32
  - Fidelity score: 0.02
  - Average score: 0.18

- Model used for summary extraction: BART
  - Rouge score: 0.22
  - BLEU score: 0.02
  - Reading score: 0.15
  - Fidelity score: 0.01
  - Average score: 0.10

- Model used for summary extraction: Vertex AI - Text Bison Text Generation model
  - Rouge score: 0.22
  - BLEU score: 0.01
  - Reading score: 0.60
  - Fidelity score: 0.01
  - Average score: 0.21

- Model used for summary extraction: Open AI - Chat GPT (GPT 3.5)
  - Rouge score: 0.48
  - BLEU score: 0.12
  - Reading score: 0.21
  - Fidelity score: 0.09
  - Average score: 0.22
    
