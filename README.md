# InstaDeep-assignment

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


## Problem with streamlit

It's commented and not working. I could not figure out how to make it work with 
vertexai as it could not recongnize vertexai one as imported, however all the code I 
wrote for using streamlit is commented and placed here:

```bash
 src/streamlit_app_breaking_commented.py
```

# Documentation

For more details of the implementation you may check "documentation.md" file