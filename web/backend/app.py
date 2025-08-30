from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import uvicorn

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from inference import MedicalQAInferenceService

# Define request/response models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    success: bool

app = FastAPI(title="Medical Chat API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the inference service
MODEL_PATH = "models/checkpoints/checkpoint_epoch_10.pt"
VOCAB_PATH = "data/processed/vocab.txt"

try:
    service = MedicalQAInferenceService(MODEL_PATH, VOCAB_PATH)
    print("Medical QA Service initialized successfully!")
except Exception as e:
    print(f"Error initializing Medical QA Service: {str(e)}")
    sys.exit(1)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.question:
            raise HTTPException(status_code=400, detail="No question provided")
        
        # Generate response using our medical QA service
        response = service.generate_response(request.question)
        
        return ChatResponse(
            answer=response,
            success=True
        )
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
