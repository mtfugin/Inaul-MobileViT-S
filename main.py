from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model and allocate tensors
try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    logger.info("Model loaded successfully")
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info(f"Input shape: {input_details[0]['shape']}")
    logger.info(f"Output shape: {output_details[0]['shape']}")
except Exception as e:
    logger.error(f"Failed to load TFLite model: {str(e)}")
    raise RuntimeError(f"Failed to load TFLite model: {str(e)}")

class InputData(BaseModel):
    input: list  # Expect a list of floats, or reshape as needed

@app.post("/predict/")
def predict(data: InputData):
    try:
        # Convert input data to NumPy array with appropriate dtype
        input_data = np.array(data.input, dtype=np.float32)

        # Reshape if needed
        input_data = input_data.reshape(input_details[0]['shape'])

        # Set tensor and invoke
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return {"prediction": output_data.tolist()}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
