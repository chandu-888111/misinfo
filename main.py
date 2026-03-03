import os
import base64
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys - In Vercel, set these in Settings > Environment Variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "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")

client = Groq(api_key=GROQ_API_KEY)

class NewsRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "TruthLens Backend is running!", "status": "online"}

@app.post("/predict")
async def verify_text(request: NewsRequest):
    try:
        search_res = requests.post("https://api.tavily.com/search", 
            json={"api_key": TAVILY_API_KEY, "query": request.text, "search_depth": "advanced"}).json()
        context = "\n".join([r['content'] for r in search_res.get("results", [])])
        
        prompt = f"Statement: {request.text}\nContext: {context}\nInstruction: Start your response with 'LABEL: REAL', 'LABEL: FAKE', or 'LABEL: AI_GENERATED'. Then provide reasoning."
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], 
            model="llama-3.3-70b-versatile"
        )
        return {"prediction": chat.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-image")
async def verify_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        b64_image = base64.b64encode(content).decode('utf-8')
        
        # Using a reliable vision model ID
        chat = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview", 
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is this image Real, Fake, or AI-generated? Start your response with 'LABEL: REAL', 'LABEL: FAKE', or 'LABEL: AI_GENERATED'. Explain the details."},
                    {"type": "image_url", "image_url": {"url": f"data:{file.content_type};base64,{b64_image}"}}
                ]
            }],
        )
        return {"prediction": chat.choices[0].message.content}
    except Exception as e:
        return {"prediction": f"Server Error: {str(e)}"}
