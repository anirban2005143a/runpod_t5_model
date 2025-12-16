import os
import torch
import runpod
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
# TOKEN-BASED CHUNKING
# -------------------------------------------------
def chunk_text_tokenwise(text, tokenizer, max_tokens=900):
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks


# -------------------------------------------------
# SUMMARY GENERATION
# -------------------------------------------------
def generate_summary(model, tokenizer, text, device, params):
    if not model or not tokenizer:
        return "Model not initialized."

    chunks = chunk_text_tokenwise(
        text,
        tokenizer,
        max_tokens=params["chunk_tokens"]
    )

    summaries = []

    for chunk in chunks:
        # Factual & legal-focused prompt
        prompt = (
            "Summarize the following legal judgment clearly and accurately. "
            "Focus only on the facts, charges, evidence, arguments, and the court's final decision. "
            "Do NOT add any assumptions, opinions, interpretations, or extra commentary. "
            "Preserve all key details such as dates, parties involved, sections of law, and outcomes.\n\n"
            f"{chunk}"
        )


        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)

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
            )

        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return "\n\n".join(summaries)


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
        "attention_mask": job_input.get("attention_mask", None)
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
