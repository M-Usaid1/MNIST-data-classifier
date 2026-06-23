from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
from model import predictor

app = FastAPI(title="MNIST Digit Recognition API")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "MNIST Digit Recognition API", "status": "running"}

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    """
    Predict digit from uploaded image
    """
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Make prediction
        result = predictor.predict(image_array)
        
        return {
            "success": True,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["all_probabilities"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}