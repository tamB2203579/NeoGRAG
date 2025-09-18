from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.readers.file import PandasExcelReader
from dotenv import load_dotenv
from uuid import uuid4
from glob import glob
from NED import applyNED
import pandas as pd
import os

from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore
from classification import classify_text

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

class GraphRAG:
    def __init__(self, model_name="gpt-4o-mini"):
        # Initialize components
        self.knowledge_graph = KnowledgeGraph()
        self.vector_store = VectorStore()
        
        # Initialize LLM
        if model_name == "gpt-4o-mini":
            self.llm = ChatOpenAI(model=model_name, temperature=0)
        else:
            self.llm = ChatMistralAI(model=model_name, temperature=0)

    def load_excel_data(self, dir="./data/processed"):
        """
        Load Excel files from a directory and combine them into a DataFrame.
        Each row gets a unique ID.
        """
        excel_files = glob(f"{dir}/*.xlsx")
        print(f"Found {len(excel_files)} Excel files in {dir}")

        all_dfs = []

        for file in excel_files:
            df = pd.read_excel(file, engine="openpyxl")
            print(f" - {file}: {len(df)} rows")
            all_dfs.append(df)

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"Combined dataset: {len(combined_df)} rows")

            combined_df["id"] = [str(uuid4()) for _ in range(len(combined_df))]
            combined_df["label"] = [classify_text(row['text']) for _, row in combined_df.iterrows()]

            return combined_df
        else:
            print("No data loaded.")
            return None
            
    def initialize_vector_store(self, data_dỉr="./data/processed"):
        """
        Initialize the vector store:
        1. Load documents from Excel files
        2. Create vector embeddings
        3. Store in vector database
        """
        print("Initializing vector store...")
        
        # Load chunks data
        documents = SimpleDirectoryReader(
            input_dir=data_dỉr,
            file_extractor={
                ".xlsx": PandasExcelReader(),
            }
        ).load_data()
        
        # Create vector store
        try:
            self.vector_store.create_vector_store(documents)
            print("Vector store initialization completed successfully!")
            return True
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return False

    def initialize_knowledge_graph(self):
        """
        Initialize the knowledge graph:
        1. Load documents from Excel files
        2. Extract entities and relationships
        3. Build knowledge graph in Neo4j
        """
        print("Initializing knowledge graph...")
        
        # Load chunks data
        df = self.load_excel_data()
        
        if df is not None:
            # Convert dataframe rows to Document objects
            documents = [Document(text=row['text'], id_=row['id'], metadata={"label": row['label']}) for _, row in df.iterrows()]
            print(f"Created {len(documents)} documents for knowledge graph building")
            
            if not documents:
                print("No documents created due to data loading issues")
                return False
            
            try:
                # Clear existing database and create constraints
                self.knowledge_graph.clear_database()
                self.knowledge_graph.create_constraints()
                
                # Build knowledge graph
                self.knowledge_graph.build_knowledge_graph(documents)
                
                print("Knowledge graph initialization completed successfully!")
                return True
            except Exception as e:
                print(f"Error creating knowledge graph: {e}")
                return False
        else:
            print("Skipping knowledge graph initialization due to missing data")
            return False
            
    def initialize_system(self):
        """
        Initialize the complete GraphRAG system:
        1. Initialize vector store
        2. Initialize knowledge graph
        """
        print("Initializing complete GraphRAG system...")
        
        vector_success = self.initialize_vector_store()
        graph_success = self.initialize_knowledge_graph()
        
        if vector_success and graph_success:
            print("Complete GraphRAG system initialization successful!")
            return True
        else:
            print("GraphRAG system initialization completed with some errors.")
            return False
        
    def chitchat_resposne(self, query):
        """
        Generate chitchat response
        """
        template = """
            You are a CTU helpful AI assistant named "REBot" that answer base on user input: {query}. Be nice and gentle in an academic way.
            Your task is to answer questions about the university’s regulations, procedures, and policies accurately and helpfully.

            Requirements:
            + Try to introduce yourself including your name, your jobs.
            + For chitchat query of the user, answer in a nicely and gently way and try to guide the user ask about CTU.
            + Avoid answer query's topics that relating to terrorism, reactionary or swear.
            + Answer in **Vietnamese**
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        response = chain.invoke({"query": query})
        return {
            "query": query,
            "response": response
        }
            
    def generate_response(self, query, senNED, label=None):
        """
        Generate a response using the GraphRAG system.
        """
        # Get vector results
        vector_results = self.vector_store.get_vector_results(query, top_k=5)
        
        # Format vector context
        vector_context = "\n\n".join([
            f"Đoạn {i+1} (Điểm tương đồng: {result['score']:.4f}):\n{result['text']}"
            for i, result in enumerate(vector_results)
        ])

        # Get graph context
        graph_context = self.knowledge_graph.get_graph_context(query, label=label)

        if not graph_context:
            graph_context = ""
        
        # Load prompt template
        with open("prompt/graphrag_query.txt", "r", encoding="utf-8") as f:
            template = f.read()
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = chain.invoke({
            "query": query,
            "vector_context": vector_context,
            "graph_context": graph_context,
            "label": label,
            "NED": senNED,
        })
        
        return {
            "query": query,
            "response": response,
            "vector_context": vector_context,
            "graph_context": graph_context
        }
