from graph_rag import GraphRAG

# Create vector store storage
graphrag = GraphRAG(model_name="gpt-4o-mini")
graphrag.initialize_vector_store()