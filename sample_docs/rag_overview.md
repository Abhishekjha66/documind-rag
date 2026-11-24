# Retrieval-Augmented Generation (RAG) Overview

Retrieval-Augmented Generation (RAG) is a technique that improves large language model (LLM) answers by retrieving relevant information from an external knowledge source at query time and adding it to the prompt. Instead of relying only on what the model memorised during training, a RAG system grounds its answer in real documents.

## Why use RAG?

- **Fewer hallucinations:** the model is instructed to answer only from retrieved passages.
- **Fresh knowledge:** new documents can be indexed without retraining the model.
- **Private data:** company or personal documents stay in your own vector store.
- **Traceability:** answers can cite the exact source passages.

## Pipeline steps

1. **Ingestion:** documents (PDF, DOCX, TXT) are parsed into plain text.
2. **Chunking:** text is split into overlapping chunks of a few hundred tokens so each chunk keeps local context.
3. **Embedding:** each chunk is converted into a dense vector using an embedding model such as BGE-small.
4. **Indexing:** vectors are stored in a vector database or an in-memory index.
5. **Retrieval:** the user question is embedded and the top-k most similar chunks are found with cosine similarity. Maximal Marginal Relevance (MMR) can diversify the results.
6. **Generation:** the retrieved chunks are placed in the prompt and the LLM writes a grounded answer with citations.

## Common failure modes

Poor chunk sizes can cut answers in half. Weak embeddings retrieve irrelevant passages. Missing instructions let the LLM answer from its own memory. Evaluating retrieval quality (recall@k) separately from generation quality is good practice.
