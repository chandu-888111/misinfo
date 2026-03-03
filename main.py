import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Allow Flutter to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenRouter Client
client = OpenAI(
    api_key="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", # Paste your exact key here
    base_url="https://openrouter.ai/api/v1"
)

class NewsRequest(BaseModel):
    text: str

@app.post("/predict")
async def verify_text(request: NewsRequest):
    try:
        # The Auto-Router handles text seamlessly
        response = client.chat.completions.create(
            model="openrouter/free", 
            messages=[{"role": "user", "content": f"Fact-check this claim: {request.text}. Verdict: [Real/Fake]. Reason:"}]
        )
        return {"prediction": response.choices[0].message.content}
    except Exception as e:
        return {"prediction": f"Text Error: {str(e)}"}

@app.post("/predict-image")
async def verify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # The Auto-Router detects the image payload and finds a compatible free vision model
        response = client.chat.completions.create(
            model="openrouter/free", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Is this image Real, Fake, or AI-generated? Explain clearly."},
                        {"type": "image_url", "image_url": {"url": f"data:{file.content_type};base64,{base64_image}"}}
                    ],
                }
            ],
        )
        return {"prediction": response.choices[0].message.content}
    except Exception as e:
        return {"prediction": f"Image Error: {str(e)}"}