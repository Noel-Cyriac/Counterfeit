import streamlit as st
import tensorflow.lite as tflite
import numpy as np
from PIL import Image

# Load the TFLite model
def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print("Expected Input Shape:", input_details[0]['shape'])  # Check the expected shape
    return interpreter


# Preprocess image
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)  # Resize to match model input size
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image



# Perform inference
def predict(interpreter, front_image, back_image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Process both images separately
    front_image = preprocess_image(front_image)
    back_image = preprocess_image(back_image)

    # Run inference on front image
    interpreter.set_tensor(input_details[0]['index'], front_image)
    interpreter.invoke()
    output_front = interpreter.get_tensor(output_details[0]['index'])

    # Run inference on back image
    interpreter.set_tensor(input_details[0]['index'], back_image)
    interpreter.invoke()
    output_back = interpreter.get_tensor(output_details[0]['index'])

    # Combine both results (Taking the average score)
    final_score = (output_front[0] + output_back[0]) / 2

    return "Genuine" if final_score > 0.5 else "Counterfeit"


# Streamlit App
st.title("Indian Currency Counterfeit Detector")

st.write("Upload front and back images of the currency note.")

front_img = st.file_uploader("Upload Front Side", type=["jpg", "png", "jpeg"])
back_img = st.file_uploader("Upload Back Side", type=["jpg", "png", "jpeg"])

if front_img and back_img:
    front_img = Image.open(front_img)
    back_img = Image.open(back_img)

    st.image([front_img, back_img], caption=["Front Side", "Back Side"], width=150)

    interpreter = load_model("currency_model.tflite")  # Provide correct path
    result = predict(interpreter, front_img, back_img)
    
    st.subheader(f"Result: {result}")
