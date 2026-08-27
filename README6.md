# Experiment 6: RETRIEVAL-AUGMENTED GENERATION (RAG) SYSTEM USING VECTOR DATABASES

## AIM

To build a Retrieval-Augmented Generation (RAG) system that retrieves relevant document chunks from a vector database and uses an LLM to generate grounded answers.

## ALGORITHM

1. Collect and split source documents into smaller text chunks.
2. Generate vector embeddings for each chunk using a sentence-transformer model.
3. Store the embeddings in a FAISS vector index.
4. Embed the incoming user query using the same embedding model.
5. Perform similarity search to retrieve the top-k most relevant chunks.
6. Construct a prompt combining the retrieved chunks and the query.
7. Pass the prompt to an LLM to generate the final grounded answer.

## PROGRAM / CODE

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# 1. Knowledge base
documents = [
    "The Eiffel Tower is located in Paris, France and was completed in 1889.",
    "Retrieval-Augmented Generation combines document retrieval with text generation.",
    "Python is a popular high-level programming language used in AI development.",
    "Vector databases store embeddings and support fast similarity search."
]

# 2. Embed documents
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents)

# 3. Build FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# 4. Query and retrieve top-2 relevant chunks
query = "What is RAG in AI?"
query_embedding = embed_model.encode([query])
D, I = index.search(np.array(query_embedding), k=2)
retrieved_chunks = [documents[i] for i in I[0]]

# 5. Build augmented prompt and generate answer
context = " ".join(retrieved_chunks)
prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

generator = pipeline("text2text-generation", model="google/flan-t5-base")
answer = generator(prompt, max_length=60)

print("Retrieved Context:", retrieved_chunks)
print("Answer:", answer[0]["generated_text"])
```

## SAMPLE INPUT

query = "What is RAG in AI?"

## SAMPLE OUTPUT

```text
Retrieved Context: ['Retrieval-Augmented Generation combines document retrieval with text generation.', 'Vector databases store embeddings and support fast similarity search.']
Answer: RAG combines document retrieval with text generation using vector databases.
```

## RESULT

A Retrieval-Augmented Generation system was successfully implemented using FAISS as the vector database, producing answers grounded in retrieved document context.
