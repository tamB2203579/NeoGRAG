from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.readers.file import PandasExcelReader
from embedding import LlamaIndexPhobertEmbedding
import logging
import faiss
import os

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, storage_dir="./storage", data_dir="./data/processed"):
        self.storage_dir = storage_dir
        self.data_dir = data_dir
        self.embed_model = LlamaIndexPhobertEmbedding()
        Settings.embed_model = self.embed_model
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
    def create_vector_store(self, docs):
        """
        Create a vector store index from document chunks and save it to disk.
        """
        # Create vector store index
        print("Creating vector index...")
        d = 768
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)

        # Save the index to disk
        print("Saving vector index to storage...")
        vector_index.storage_context.persist(persist_dir=self.storage_dir)
        print("Vector store created and saved successfully!")
        
        return vector_index
        
    def load_vector_store(self):
        """
        Load a previously saved vector store index from disk.
        """
        if not os.path.exists(self.storage_dir):
            raise FileNotFoundError(f"Storage directory '{self.storage_dir}' does not exist.")
            
        # Check if the directory is not empty before trying to load
        if os.listdir(self.storage_dir):
            logger.info(f"Loading vector store from {self.storage_dir}")
            vector_store = FaissVectorStore.from_persist_dir(self.storage_dir)
            storage_context = StorageContext.from_defaults(
                persist_dir=self.storage_dir, vector_store=vector_store
            )
            loaded_index = load_index_from_storage(storage_context=storage_context)
            return loaded_index
        else:
            logger.info("Creating new vector store")
            docs = SimpleDirectoryReader(
                input_dir=self.data_dir,
                file_extractor={
                    ".xlsx": PandasExcelReader(),
                }
            ).load_data()
            index = self.create_vector_store(docs)
            return index
        
    def get_vector_results(self, query, top_k=5):
        """
        Get relevant results from the vector store based on the query.
        """
        index = self.load_vector_store()
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        results = []
        for i, node in enumerate(nodes):
            results.append({
                "text": node.node.text,
                "score": node.score,
                "id": node.node.id_,
                "metadata": node.node.metadata
            })
        
        return results
    
    def update_vector_store(self, docs):
        """
        Add new documents to existing vector store.
        """
        index = self.load_vector_store()
            
        for doc in docs:
            index.insert(doc)
            
        # Save the updated index
        index.storage_context.persist(persist_dir=self.storage_dir)
        return index
