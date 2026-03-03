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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "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")

client = Groq(api_key=GROQ_API_KEY)

class NewsRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "TruthLens API is Online"}

@app.post("/predict")
async def verify_text(request: NewsRequest):
    try:
        search_res = requests.post("https://api.tavily.com/search", 
            json={"api_key": TAVILY_API_KEY, "query": request.text, "search_depth": "advanced"}).json()
        context = "\n".join([r['content'] for r in search_res.get("results", [])])
        
        # PROMPT UPDATE: Requesting a detailed point-by-point breakdown
        prompt = f"""Statement to check: "{request.text}"
Context from Web: {context}

INSTRUCTIONS:
You are a highly analytical forensic fact-checker. 
1. Evaluate the statement. Start your response EXACTLY with 'LABEL: REAL', 'LABEL: FAKE', or 'LABEL: AI_GENERATED'.
2. After the label, provide a DETAILED, POINT-BY-POINT report explaining your verdict. 
3. Use bullet points or numbered lists to break down the context, evidence, and logical deductions step-by-step."""

        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], 
            model="llama-3.3-70b-versatile",
            temperature=0.1 
        )
        return {"prediction": chat.choices[0].message.content}
    except Exception as e:
        return {"prediction": f"LABEL: ERROR\nBackend Error: {str(e)}"}

@app.post("/predict-image")
async def verify_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        b64_image = base64.b64encode(content).decode('utf-8')
        
        # PROMPT UPDATE: Requesting forensic visual analysis in points
        chat = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image meticulously. Start your response EXACTLY with 'LABEL: REAL', 'LABEL: FAKE', or 'LABEL: AI_GENERATED'. Then, provide a DETAILED, POINT-BY-POINT analytical report explaining the visual evidence, lighting, artifacts, or context that led to your verdict. Use bullet points."},
                    {"type": "image_url", "image_url": {"url": f"data:{file.content_type};base64,{b64_image}"}}
                ]
            }],
            temperature=0.2
        )
        return {"prediction": chat.choices[0].message.content}
    except Exception as e:
        return {"prediction": f"LABEL: ERROR\nVision Error: {str(e)}"}
