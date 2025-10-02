# Inaul-MobileViT-S (TensorFlow Lite)

This repository contains **inaul-mobilevit-s.tflite**, a TensorFlow Lite adaptation of the [MobileViT-S](https://github.com/apple/ml-cvnets) architecture.  
It is designed for efficient mobile and edge-device deployment.

---

## ✨ Features
- Based on **MobileViT-S** (Apache 2.0 License).
- Optimized and converted for **TensorFlow Lite** format.
- Lightweight and mobile-friendly.
- Easy integration into Android and embedded applications.

---

## 📌 Model Details
- **File**: `inaul-mobilevit-s.tflite`  
- **Framework**: TensorFlow Lite  
- **Base Architecture**: MobileViT-S  
- **Creator/Modifier**: Amerogin Kamid (2025)  

---

## 📜 License
This project is licensed under the [Apache License 2.0](LICENSE).

### Attribution
This work is derived from **MobileViT-S**, which is licensed under the Apache License 2.0.  
See [NOTICE](NOTICE) for attribution details.

---

## 🚀 Usage
Clone this repo and import the model into your TensorFlow Lite workflow:

```python
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="inaul-mobilevit-s.tflite")
interpreter.allocate_tensors()
