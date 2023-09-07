
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Appname
st.set_page_config(page_title="Medical Waste Classifier", layout="wide")

st.markdown("<h1 style='text-align: center; color: #fff;'>Medical Waste Classifier</h1>", unsafe_allow_html=True)

# Load your model and its weights
model = tf.keras.models.load_model('hackomedfinaaaal.h5')
class_names = ["Medical Hazardous Waste", "Inorganic Waste", "Organic Waste"]  # List of your class names

# Define the Streamlit app
def main():
    st.write("Upload an image for classification")
    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    if run: 
        camera = cv2.VideoCapture(1)

        while run:
            _, frame = camera.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(image)

            resized_image = cv2.resize(image, (224, 224))  # Resize to your model's input size
            image_array = img_to_array(resized_image)
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            image_array = preprocess_input(image_array)  # Preprocess using the same function as during training

            # Make a prediction
            predictions = model.predict(image_array)
            predicted_class = np.argmax(predictions[0])  # Assuming your model outputs class probabilities

            # Print the prediction
            predicted_probability = predictions[0][predicted_class]
            if np.isnan(predicted_probability).any():
                st.write("No Waste Detected(Default)")
            else:
                st.write(f"Predicted class: {class_names[predicted_class]}")

                st.write(f"Probability: {predicted_probability*100:.2f}%")
              # Probability of the predicted class


            # Print the probability
           
        
        st.write('Stopped')
        camera.release()  # Release the camera when the loop is stopped

    items = [
        'Organic Waste', 'Inorganic Waste', 'Medical Hazardous Waste'
    ]

    st.title("This model is capable of classifying:")
    for item in items:
        st.write("- " + item)

# Run the app
if __name__ == '__main__':
    main()
