from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from PIL import Image
from google.genai import types
from io import BytesIO
import base64
import os
import json

# Load environment variables
load_dotenv()

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in environment variables.")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    prompt:str

@app.post("/generate-text")
async def generate_text(request: TextRequest):
    contents = request.prompt
    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=contents 
    )
    print(response.text)
    return response.text



class ImageRequest(BaseModel):
    prompt: str  # Changed from 'contents' to match common practice

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    contents = request.prompt

    try:
        # Generate content
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
              response_modalities=['Text', 'Image']
            )
        )
                # Check for images in response
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO((part.inline_data.data)))

                # save the image
                image.save('gemini-native-image.png')

                #read the image
                with open('gemini-native-image.png', 'rb') as image_file:
                    image_data = image_file.read()
                    image_str = base64.b64encode(image_data).decode("utf-8")     
                    return {"image": image_str}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# edit image 
class EditImageRequest(BaseModel):
    prompt: str
    image: str  # Base64 encoded image string

@app.post("/edit-image")
async def edit_image(request: EditImageRequest):
    text_input = request.prompt
    image_data = base64.b64decode(request.image)
    image = Image.open(BytesIO(image_data))

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[text_input, image],
            config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
        )

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)  # Optional: Print text output
            elif part.inline_data is not None:
                # Convert inline image data to base64 without saving
                edited_image = Image.open(BytesIO(part.inline_data.data))
                buffered = BytesIO()
                edited_image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                return {"image": image_base64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/get-trends")
async def get_trends():
    print("Getting trends")
    trends = json.load(open('trending.json'))
    return trends





# run the app: uvicorn app:app --reload