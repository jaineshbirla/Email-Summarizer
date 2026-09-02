from fastapi import FastAPI
from summarizer import Summarizer



app = FastAPI()

@app.get("/")
def hello():
    return {"Greet" : "Hello"}

@app.get("/summarize")
def summarize_endpoint(req):
    
    return Summarizer.summarize_text(req.text, req.max_length, req.min_length)
    
    
    