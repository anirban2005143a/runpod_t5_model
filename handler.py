import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- GLOBAL MODEL/TOKENIZER INITIALIZATION ---
# Load once when the worker starts (cold start)
try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TOKENIZER = AutoTokenizer.from_pretrained("AnirbanDas2005/PnHLayman")
    MODEL = AutoModelForSeq2SeqLM.from_pretrained("AnirbanDas2005/PnHLayman")
    MODEL.to(DEVICE)
    print(f"Model loaded successfully on device: {DEVICE}")
except Exception as e:
    print(f"Error loading model: {e}")
    TOKENIZER, MODEL, DEVICE = None, None, "cpu"


# --- MODEL LOGIC FUNCTIONS ---

def chunk_text(text, max_length=1024):
    """Splits text into word-based chunks of maximum length."""
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)
        if len(chunk) > max_length:
            chunks.append(' '.join(chunk[:-1]))
            chunk = [chunk[-1]]
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

def generate_summary(model, tokenizer, case_text, device, params):
    """
    Generates the summary using customizable generation parameters.
    'params' is a dictionary containing the generation settings.
    """
    if not model or not tokenizer:
        return "Model not initialized."
        
    chunks = chunk_text(case_text, max_length=params.get('max_input_length', 1024))
    all_summaries = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="longest", max_length=params.get('max_input_length', 1024))
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Generator settings are dynamically pulled from the 'params' dictionary
        summary_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=params.get('max_new_tokens', 128),
            num_beams=params.get('num_beams', 8),
            early_stopping=params.get('early_stopping', True),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=params.get('no_repeat_ngram_size', 3),
            length_penalty=params.get('length_penalty', 0.8)
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        all_summaries.append(summary)

    return " ".join(all_summaries)


# --- RUNPOD HANDLER FUNCTION ---

def handler(job):
    """
    The main handler function for the RunPod Serverless worker.
    The 'job' object contains the input data and optional parameters.
    """
    print(f"Received job: {job['id']}")
    
    if not MODEL or not TOKENIZER:
        return {"error": "Model failed to load. Check logs."}
        
    try:
        # 1. Get the required input text
        job_input = job['input']
        case_text = job_input['text']
    except KeyError:
        return {"error": "Input payload must contain a 'text' key with the judgment content."}
    
    if not case_text:
        return {"error": "Input text cannot be empty."}

    # 2. Extract and compile generation parameters, using defaults if not provided
    generation_params = {
        # Default values from your original code are set here:
        "max_new_tokens": job_input.get("max_new_tokens", 128),
        "num_beams": job_input.get("num_beams", 8),
        "early_stopping": job_input.get("early_stopping", True),
        "no_repeat_ngram_size": job_input.get("no_repeat_ngram_size", 3),
        "length_penalty": job_input.get("length_penalty", 0.8),
        "max_input_length": job_input.get("max_input_length", 1024) # Used for chunking/tokenization
    }

    # 3. Generate the summary
    try:
        summary = generate_summary(MODEL, TOKENIZER, case_text, DEVICE, generation_params)
        
        # Return the result
        return {
            "summary": summary,
            "input_text_length": len(case_text),
            "parameters_used": generation_params # For confirmation
        }
    
    except Exception as e:
        print(f"Error during summary generation: {e}")
        return {"error": f"An error occurred during processing: {str(e)}"}
