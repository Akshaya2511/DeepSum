from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
import torch
from pydantic import BaseModel
from summarizer import Summarizer
from deep_translator import GoogleTranslator
from qa_model import router as qa_router

app = FastAPI(title="DeepSum - Summarization, QA & Translation")

# Allow frontend JS to work locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(qa_router)
'''
# Load Summarizers
class Summarizer:
    def __init__(self, model_type="t5"):
        self.device = "cpu"
        if model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-large").to(self.device)
        else:
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
            self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(self.device)

    def summarize(self, text, max_length=1500, min_length=50):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=min_length, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    def summarize(self, text):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt",max_length=1024,truncation=True).to(self.device)

        input_length = inputs.shape[1]

        # Auto-adjust based on input length
        max_length = min(512, int(input_length * 0.5))     # up to 50% of input tokens
        min_length = max(30, int(max_length * 0.5))         # at least 50% of max

        summary_ids = self.model.generate(inputs,max_length=max_length,min_length=min_length,num_beams=4,early_stopping=True)

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

t5_summarizer = Summarizer("t5")
bart_summarizer = Summarizer("bart")

# Request Models
class SummarizeRequest(BaseModel):
    text: str
    model: str = "t5"

class TranslateRequest(BaseModel):
    text: str
    target_lang: str

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("1.html", {"request": request})

@app.post("/summarize/")
async def summarize(request: SummarizeRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required for summarization.")

    summarizer = t5_summarizer if request.model == "t5" else bart_summarizer
    summary = summarizer.summarize(request.text)
    return {"summary": summary}

@app.post("/translate/")
async def translate(req: TranslateRequest):
    if not req.text or not req.target_lang:
        raise HTTPException(status_code=400, detail="Both text and target language required.")

    translated = GoogleTranslator(source="auto", target=req.target_lang).translate(req.text)
    return {"translated_text": translated}

'''

# Load T5
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load BART
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")



# Helper functions
def split_text(text, max_chunk_size=1024):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1
        if current_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_t5(text):
    input_ids = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = t5_model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Summarization logic using BART
def summarize_bart(text):
    input_ids = bart_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(input_ids, max_length=150, min_length=40, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Summarization for long text
def summarize_long_text(text: str, method: str = "t5"):
    chunks = split_text(text)
    summaries = []

    for chunk in chunks:
        if method == "t5":
            summaries.append(summarize_t5(chunk))
        elif method == "bart":
            summaries.append(summarize_bart(chunk))
        else:
            raise ValueError("Unsupported model.")

    return " ".join(summaries)

# Web form routes
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "summary": "",
        "selected_model": "t5"
    })

@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request):
    form_data = await request.form()
    input_text = form_data.get("input_text")
    model_choice = form_data.get("model_choice", "t5")

    if not input_text:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "summary": "No text provided.",
            "selected_model": model_choice
        })

    summary = summarize_long_text(input_text, method=model_choice)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "summary": summary,
        "original": input_text,
        "selected_model": model_choice
    })

# API request models
class SummarizeRequest(BaseModel):
    text: str
    model: str = "t5"

class TranslateRequest(BaseModel):
    text: str
    target_lang: str

# API endpoints
@app.post("/summarize/")
async def summarize(req: SummarizeRequest):
    if not req.text or req.model not in ["t5", "bart"]:
        raise HTTPException(status_code=400, detail="Invalid input or model not supported.")
    summary = summarize_long_text(req.text, method=req.model)
    return {"summary": summary}

@app.post("/translate/")
async def translate(req: TranslateRequest):
    if not req.text or not req.target_lang:
        raise HTTPException(status_code=400, detail="Both text and target language required.")
    translated = GoogleTranslator(source="auto", target=req.target_lang).translate(req.text)
    return {"translated_text": translated}
