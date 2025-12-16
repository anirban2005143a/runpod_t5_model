import os
import torch
import runpod
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------------------------
# Hugging Face cache location
# -------------------------------------------------
os.environ["HF_HOME"] = "/runpod-volume/huggingface-cache"

MODEL_ID = "AnirbanDas2005/PnHLayman"

# -------------------------------------------------
# GLOBAL MODEL LOAD
# -------------------------------------------------
try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
    MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    MODEL.to(DEVICE)
    MODEL.eval()

    print(f"✅ Model loaded on {DEVICE}")

except Exception as e:
    print(f"❌ Model load failed: {e}")
    MODEL = None
    TOKENIZER = None
    DEVICE = "cpu"


# -------------------------------------------------
# LANGCHAIN CHUNKING (RecursiveCharacterTextSplitter)
# -------------------------------------------------
def chunk_text(text, tokenizer, max_tokens=900, overlap_tokens=50):
    """
    Chunks text using LangChain's RecursiveCharacterTextSplitter,
    configured to respect legal document structure (paragraphs, sentences)
    and use the HuggingFace tokenizer's length function for accurate token counting.
    """
    
    # 1. Define the separators for legal documents: try to split by paragraph, then newline, then sentence.
    # We use a hierarchical approach, favoring larger chunks first.
    separators = [
        "\n\n",   # Double newline (paragraph/section break)
        "\n",     # Single newline
        ". ",     # Sentence end
        # " ",      # Whitespace (last resort)
    ]

    # 2. Define a function to calculate chunk length in tokens (using the actual model tokenizer)
    def token_length_function(chunk):
        return len(tokenizer.encode(chunk))

    # 3. Instantiate the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        length_function=token_length_function, # Use the token-based length function
        is_separator_regex=False,
    )

    # 4. Split the text
    chunks = text_splitter.split_text(text)
    
    # Optional check to ensure all chunks respect the max_tokens limit
    # (The splitter handles this internally, but it's good practice for debugging)
    # for i, chunk in enumerate(chunks):
    #     if token_length_function(chunk) > max_tokens:
    #         print(f"Warning: Chunk {i} size ({token_length_function(chunk)}) exceeds max_tokens.")

    return chunks


# -------------------------------------------------
# TRIM TO LAFT FULLSTOP
# -------------------------------------------------
def trim_to_last_fullstop(text: str) -> str:
    """Trim text to the last full stop (.) if it exists."""
    last_dot_index = text.rfind(".")
    if last_dot_index != -1:
        return text[:last_dot_index + 1]
    return text


# -------------------------------------------------
# SUMMARY GENERATION
# -------------------------------------------------
def generate_summary(model, tokenizer, text, device, params):
    if not model or not tokenizer:
        return "Model not initialized."

    # Use the new LangChain chunking function
    chunks = chunk_text(
        text,
        tokenizer,
        max_tokens=params["chunk_tokens"],
        overlap_tokens=params.get("chunk_overlap", 50) 
    )

    summaries = []

    for chunk in chunks:
        # **ENHANCED PROMPT**
        # prompt = (
        #     "Summarize the following portion of a legal judgment in clear, factual language. "
        #     "Focus only on: facts of the case, parties involved, dates and timeline of events, "
        #     "charges, evidence presented, arguments of prosecution and defense, and the court's findings or decisions. "
        #     "Do NOT add assumptions, opinions, or commentary. Preserve all important factual details and legal references.\n\n"
        #     f"{chunk}"
        # )

        # Tokenize and move inputs to device
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate summary
        with torch.no_grad():
            output_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=params["max_new_tokens"],
                num_beams=params["num_beams"],
                early_stopping=params["early_stopping"],
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=params["no_repeat_ngram_size"],
                length_penalty=params["length_penalty"]
                no_repeat_ngram_size=3,
                do_sample=False,
            )

        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        summary = trim_to_last_fullstop(summary)
        summaries.append(summary)

    # Merge summaries
    final_summary = " ".join(summaries)
    return {"generated_text": final_summary}



# -------------------------------------------------
# RUNPOD HANDLER
# -------------------------------------------------
def handler(job):
    print(f"Received job: {job['id']}")

    if not MODEL or not TOKENIZER:
        return {"error": "Model not loaded."}

    try:
        job_input = job["input"]
        case_text = job_input["text"]
    except KeyError:
        return {"error": "Missing required field: 'text'"}

    if not case_text.strip():
        return {"error": "Input text cannot be empty."}

    # Optional parameters with defaults
    generation_params = {
        "max_new_tokens": job_input.get("max_new_tokens", 128),
        "num_beams": job_input.get("num_beams", 8),
        "early_stopping": job_input.get("early_stopping", True),
        "no_repeat_ngram_size": job_input.get("no_repeat_ngram_size", 3),
        "length_penalty": job_input.get("length_penalty", 0.8),
        "chunk_tokens": job_input.get("chunk_tokens", 900),
        # New parameter for chunk overlap
        "chunk_overlap": job_input.get("chunk_overlap", 50) 
    }

    try:
        summary = generate_summary(
            MODEL,
            TOKENIZER,
            case_text,
            DEVICE,
            generation_params
        )

        return {
            "summary": summary,
            "input_text_length": len(case_text),
            "parameters_used": generation_params
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
