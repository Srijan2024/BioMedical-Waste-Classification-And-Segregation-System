
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Appname
st.set_page_config(page_title="Medical Waste Classifier", layout="wide")

st.markdown("<h1 style='text-align: center; color: #fff;'>Medical Waste Classifier</h1>", unsafe_allow_html=True)

# Load your model and its weights
model = tf.keras.models.load_model('hackomedfinaaaal.h5')
class_names = ["Medical Hazardous Waste", "inorganic", "organic"]  # List of your class names

# Define the Streamlit app
def main():
    st.write("Upload an image for classification")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Resize the image
        image = image.resize((224, 224))  # Resize to your model's input size

        # Convert the image to an array and preprocess it
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = preprocess_input(image_array)  # Preprocess using the same function as during training

        # Make a prediction
        predictions = model.predict(image_array)

        # Interpret the prediction (adjust this based on your model's output)
        predicted_class = np.argmax(predictions[0])  # Assuming your model outputs class probabilities

        # Print the prediction
        
        predicted_probability = predictions[0][predicted_class]  # Probability of the predicted class

    # Print the prediction and probability
        st.write(f"Predicted class: {class_names[predicted_class]}")
        st.write(f"Probability: {predicted_probability:.4f}")        

    items = [
        'organic','inorganic','hazardous'
    ]

    st.title("This model is capable of classifying:")
    for item in items:
        st.write("- " + item)

# Run the app
if __name__ == '__main__':
    main()