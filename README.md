#chromadb-orm

requirements
- document ingestion pipeline
    - process (convert to text)
    - summary
    - convert to embeddings
    - store it to a document_db
    - delete it from a document_db

- collection management pipeline
    - custom metadata per collection
        - update metadata
        - summary of summary
    - similarity score
        - step 1 : centroid embedding of a collection
        - step 2 : top-k chunk similarity score
