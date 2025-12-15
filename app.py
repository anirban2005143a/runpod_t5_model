import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

# ----------------------------------------------------
# 1. Initialization (Runs ONCE when the worker starts)
# ----------------------------------------------------

# !!! YOUR SPECIFIED MODEL ID !!!
MODEL_ID = "AnirbanDas2005/PnHLayman" 

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

try:
    # Use AutoModelForSeq2SeqLM for T5 models
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.to(device).eval() 
    
    print(f"Model {MODEL_ID} loaded successfully onto {device}.")
    
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize model or tokenizer: {e}")
    # Model will be set to None, causing the /health check to fail

# Initialize FastAPI application
app = FastAPI()

# Pydantic schema for the request body
class Query(BaseModel):
    inputs: str
    max_new_tokens: int = 128 
    num_beams: int = 4
    chunk_max_tokens: int = 1024 
    chunk_overlap: int = 100 

# ----------------------------------------------------
# 2. Token-Aware Chunking Logic 
# ----------------------------------------------------

def chunk_text_by_tokens(text: str, max_tokens: int, overlap: int) -> List[str]:
    """
    Chunks text using a RecursiveCharacterTextSplitter based on token count
    to ensure chunk sizes are safe for the LLM context window.
    """
    if tokenizer is None:
        raise RuntimeError("Tokenizer is not loaded for chunking.")
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        # Use the model's tokenizer to measure length
        length_function=lambda t: len(tokenizer.encode(t)), 
        separators=["\n\n", "\n", " ", ""]
    )
    
    return splitter.split_text(text)


# ----------------------------------------------------
# 3. API Endpoint Logic
# ----------------------------------------------------

@app.get("/health")
def health_check():
    """Endpoint for the load balancer to check worker status."""
    if model is not None and tokenizer is not None:
        return {"status": "READY", "model": MODEL_ID, "device": device}
    else:
        # 503 Service Unavailable if the model failed to load
        raise HTTPException(
            status_code=503, 
            detail={"status": "INITIALIZING_FAILED", "error": "Model not loaded"}
        )

@app.post("/")
def process_text(query: Query):
    """Main endpoint for chunking and summarizing text."""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")
        
    case_text = query.inputs
    
    # 1. Chunk the input text
    try:
        chunks = chunk_text_by_tokens(
            case_text, 
            max_tokens=query.chunk_max_tokens,
            overlap=query.chunk_overlap
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chunking failed: {e}")
        
    all_summaries = []

    # 2. Generate summary for each chunk (Map step)
    for i, chunk in enumerate(chunks):
        
        inputs = tokenizer(
            chunk, 
            return_tensors="pt", 
            truncation=True, 
            padding="longest", 
            max_length=query.chunk_max_tokens
        )
        
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad(): 
            summary_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=query.max_new_tokens,
                num_beams=query.num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                length_penalty=0.8
            )
            
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        all_summaries.append(summary)

    # 3. Concatenate and return the final summary (Reduce step)
    return {"summary": " ".join(all_summaries), "chunk_count": len(chunks)}