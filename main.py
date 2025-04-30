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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS (restrict to your frontend in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to ["https://your-frontend-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log environment details
model_path = "mobilevit_s_float16.tflite"
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Model file exists: {os.path.exists(model_path)}")
logger.info(f"Model file size: {os.path.getsize(model_path) if os.path.exists(model_path) else 0} bytes")

# Load TFLite model and allocate tensors
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    logger.info("Model loaded successfully")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info(f"Input shape: {input_details[0]['shape']}")
    logger.info(f"Output shape: {output_details[0]['shape']}")
except Exception as e:
    logger.error(f"Failed to load TFLite model: {str(e)}")
    raise RuntimeError(f"Failed to load TFLite model: {str(e)}")

class InputData(BaseModel):
    image: str  # Base64-encoded image

# Softmax function to convert logits to probabilities
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Decode base64 image
        img_data = base64.b64decode(data.image.split(',')[1])
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img = img.resize((256, 256), Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)

        # Normalize (MobileViT: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
        img_array = (img_array - mean) / std

        # Add batch dimension: [1, 256, 256, 3]
        input_data = np.expand_dims(img_array, axis=0)

        # Verify input shape
        expected_input_shape = input_details[0]['shape']
        if tuple(input_data.shape) != tuple(expected_input_shape):
            raise ValueError(f"Input shape {input_data.shape} does not match model input {expected_input_shape}")

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        logger.info(f"Raw prediction: {output_data.tolist()}")

        # Flatten and apply softmax
        logits = output_data[0]
        probabilities = softmax(logits)
        logger.info(f"Probabilities: {probabilities.tolist()}")

        return {"prediction": probabilities.tolist()}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
