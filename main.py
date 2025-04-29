from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
        raise HTTPException(status_code=400, detail=str(e))
