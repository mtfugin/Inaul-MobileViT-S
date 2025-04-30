from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import logging
import os
from PIL import Image
import io
import base64
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS (restrict to your frontend in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to ["https://your-frontend-domain.com"] for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path
model_path = "mobilevit_s_float16.tflite"

# Log environment details
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Model file exists: {os.path.exists(model_path)}")
logger.info(f"Model file size: {os.path.getsize(model_path) if os.path.exists(model_path) else 0} bytes")

# Load TFLite model and allocate tensors
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    logger.info("Model loaded successfully")
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    logger.info(f"Input details: {input_details}")
    logger.info(f"Output details: {output_details}")
except Exception as e:
    logger.error(f"Failed to load TFLite model: {str(e)}")
    raise RuntimeError(f"Failed to load TFLite model: {str(e)}")

class InputData(BaseModel):
    image: str  # Base64-encoded image

# Softmax function to convert logits to probabilities
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

@app.get("/")
async def root():
    return {"message": "Inaul Pattern Recognition API is running"}

@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Decode base64 image
        base64_data = data.image
        
        # Check if the image includes the data URL prefix and remove if present
        if "," in base64_data:
            base64_data = re.sub(r'^data:image/[a-zA-Z]+;base64,', '', base64_data)
        
        # Decode base64 string
        img_data = base64.b64decode(base64_data)
        
        # Open image
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if needed (handles PNG with alpha channel)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to required dimensions (256x256)
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and ensure correct dtype
        img_array = np.array(img, dtype=np.float32)
        
        # Log image shape
        logger.info(f"Image shape after processing: {img_array.shape}")
        
        # Normalize using MobileViT normalization parameters
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
        
        # Apply normalization - ensure shape compatibility
        img_array = (img_array - mean) / std
        
        # Add batch dimension
        input_data = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        # Log input shape
        logger.info(f"Final input shape: {input_data.shape}")
        
        # Verify input shape matches expected shape
        expected_shape = tuple(input_details[0]['shape'])
        if input_data.shape != expected_shape:
            logger.warning(f"Input shape {input_data.shape} does not match expected {expected_shape}")
            # Reshape if needed
            input_data = np.reshape(input_data, expected_shape)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Log raw output shape
        logger.info(f"Raw output shape: {output_data.shape}")
        
        # Flatten if needed and apply softmax
        if len(output_data.shape) > 1:
            logits = output_data[0]  # Get first batch result
        else:
            logits = output_data
            
        # Convert to float64 to avoid precision issues
        logits = logits.astype(np.float64)
        
        # Apply softmax to get probabilities
        probabilities = softmax(logits)
        
        # Return top probabilities
        return {"prediction": probabilities.tolist()}
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
