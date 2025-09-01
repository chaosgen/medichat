from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sys
import os
import uvicorn

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from rag_inference import RAGMedicalQA
from config_loader import config

# Define request/response models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    context: str
    success: bool

app = FastAPI(title="RAG-Enhanced Medical Chat API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG service
try:
    service = RAGMedicalQA()
    
    # Load and index training data
    print("Loading training data for indexing...")
    df = pd.read_csv(config.get_path('data', 'raw_data_path'))
    service.index_documents(df)
    
    print("RAG Medical QA Service initialized successfully!")
except Exception as e:
    print(f"Error initializing RAG Medical QA Service: {str(e)}")
    sys.exit(1)

@app.post("/api/rag-chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.question:
            raise HTTPException(status_code=400, detail="No question provided")
        
        # Generate response using RAG-enhanced service
        response = service.generate_response(request.question)
        
        return ChatResponse(
            answer=response['answer'],
            context=response['context'],
            success=True
        )
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(
        "rag_app:app",
        host=config.api['host'],
        port=config.api['port'] + 1,  # Run on port 5001 to avoid conflict
        reload=True
    )
