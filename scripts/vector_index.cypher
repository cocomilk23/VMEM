CREATE VECTOR INDEX memory_embedding_index IF NOT EXISTS
FOR ()-[r:MEMORY]-()
ON (r.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
};
