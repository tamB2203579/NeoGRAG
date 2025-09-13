import os
import shutil
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Document
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

from collect_data import convert_pdf_to_md, process_md_files
from classification import classify_text, classify_binary
from NED import applyNED
from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore
from graph_rag import GraphRAG

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GraphRAG instance
graphrag = GraphRAG(model_name="gpt-4o-mini")
responses: List[Dict] = []

class BaseResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime = datetime.now()

class ResponseData(BaseModel):
    query: str
    response: str = ""

class AskQuestionResponse(BaseResponse):
    data: ResponseData

class UploadResponse(BaseModel):
    filename: str
    status: str
    message: str

class RetrainResponse(BaseResponse):
    processed_file: int
    processing_time: Optional[float] = None
    details: Optional[Dict]  = None

def answer_query(query: str) -> str:
    binary_label = classify_binary(query).replace("__label__", "")
    if binary_label == "chitchat":
        result = graphrag.chitchat_resposne(query)
        return result["response"]
    elif binary_label == "Academic":
        label = classify_text(query)
        senNED = applyNED(query)
        result = graphrag.generate_response(query=query, senNED=senNED, label=label)
        return result["response"]

@app.post("/ask", response_model=AskQuestionResponse)
async def ask_question(data: ResponseData):
    try:
        response_text = answer_query(data.query)

        result = ResponseData(
            query = data.query,
            response = response_text,
        )

        responses.append(result)
        return AskQuestionResponse(
            status="success",
            message="Query processed successfully",
            data=result
        )
    except Exception as e:
        logging.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed {str(e)}")

@app.post("/upload", response_model=List[UploadResponse])
async def upload_pdfs(files: List[UploadFile] = File(...)):
    responses = []
    try:
        upload_dir = "data/uploaded/raw"
        os.makedirs(upload_dir, exist_ok=True)

        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                responses.append(UploadResponse(
                    filename=file.filename,
                    status="failed",
                    message="Only PDF files are allowed"
                ))
                continue

            upload_path = os.path.join(upload_dir, file.filename)

            if os.path.exists(upload_path):
                responses.append(UploadResponse(
                    filename=file.filename,
                    status="failed",
                    message="File already exists"
                ))
                continue

            try:
                with open(upload_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                responses.append(UploadResponse(
                    filename=file.filename,
                    status="success",
                    message="PDF uploaded successfully!"
                ))
            except Exception as e:
                logging.error(f"Failed to upload {file.filename}: {str(e)}")
                responses.append(UploadResponse(
                    filename=file.filename,
                    status="failed",
                    message=f"Upload failed: {str(e)}"
                ))

        return responses
    except Exception as e:
        logging.error(f"Batch upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")

@app.post("/train", response_model=RetrainResponse)
async def retrain():
    import time
    start_time = time.time()

    try:
        upload_dir = "data/uploaded/raw"
        save_dir = "data/uploaded/processed"

        if not os.path.exists(upload_dir) or not os.listdir(upload_dir):
            raise HTTPException(status_code=400, detail="No files found to process")

        os.makedirs(save_dir, exist_ok=True)
        file_count = len(os.listdir(upload_dir))

        convert_pdf_to_md(upload_dir, save_dir)
        process_md_files(save_dir,use_semantic=True)
        
        kg = KnowledgeGraph()
        vt = VectorStore()

        df = graphrag.load_excel_data(save_dir)
        
        vt_docs_inserted = 0
        kg_chunks_processed = 0

        if df is not None and not df.empty:
            documents = [Document(text=row['text'], id_=row['id'], metadata={"label": row['label']}) for _, row in df.iterrows()]
            index = vt.load_vector_store() 
            for doc in documents:
                index.insert(doc)
            vt_docs_inserted = len(documents)
            kg.build_knowledge_graph(documents)
            kg_chunks_processed = len(documents)
        
        processing_time = time.time() - start_time

        return RetrainResponse(
            status="success",
            message="Training completed successfully",
            processed_file=file_count,
            processing_time=round(processing_time, 2),
            details={
                "vector_documents_inserted": vt_docs_inserted,
                "knowledge_graph_chunks_processed": kg_chunks_processed,
                "upload_directory": upload_dir,
                "processed_directory": save_dir
            }
        )
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"Trainning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    finally:
        shutil.rmtree(upload_dir)
        shutil.rmtree(save_dir)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="api:app", host="localhost", port=8000)