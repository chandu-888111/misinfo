import base64
import time
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Explicitly defining the app root for Serverless
app = FastAPI()

# VERY IMPORTANT: Allows your Vercel URL to accept requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenRouter/SiliconFlow Client
client = OpenAI(
    api_key="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", # Paste your exact key here
    base_url="https://api.siliconflow.cn/v1" 
)

class NewsRequest(BaseModel):
    text: str

@app.post("/predict")
async def verify_text(request: NewsRequest):
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct", 
            messages=[{"role": "user", "content": f"Fact-check this: {request.text}. Verdict: [Real/Fake]. Reason:"}],
            timeout=30.0
        )
        return {"prediction": response.choices[0].message.content}
    except Exception as e:
        return {"prediction": f"Text Analysis Error: {str(e)}"}

@app.post("/predict-image")
async def verify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Cold Start Retry Logic ensures the server doesn't timeout immediately
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="deepseek-ai/deepseek-vl2", 
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Is this image Real, Fake, or AI-generated? Explain in detail."},
                                {"type": "image_url", "image_url": {"url": f"data:{file.content_type};base64,{base64_image}"}}
                            ],
                        }
                    ],
                    timeout=60.0 
                )
                return {"prediction": response.choices[0].message.content}
            
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return {"prediction": "The AI model is currently booting up (Cold Start). Please try again in a moment."}
                time.sleep(2) 
                
    except Exception as e:
        return {"prediction": f"Image Upload Error: {str(e)}"}
