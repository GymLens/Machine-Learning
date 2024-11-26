import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

# Path ke model yang sudah dilatih
MODEL_PATH = 'gym_model_9875_12.h5'  # Ganti sesuai path model Anda

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

class_names = ['Bench Press', 'Dip Bar', 'Dumbells', 'Elliptical Machine', 'KettleBell', 'Lat Pulldown', 'Leg Press Machine', 'PullBar', 'Recumbent Bike', 'Stair Climber', 'Swiss Ball', 'Treadmill']

# Fungsi untuk preprocess gambar
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image to the required input size and normalize.
    """
    image = image.resize(target_size)  # Resize ke ukuran input model
    image_array = img_to_array(image) / 255.0  # Normalisasi
    image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch
    return image_array

# Header aplikasi
st.title("Gym Equipment Classifier")
st.write("Upload an image to classify gym equipment into one of the predefined categories.")

# Upload gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Jika ada gambar yang di-upload
if uploaded_file is not None:
    # Menampilkan gambar yang di-upload
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocessing gambar
    preprocessed_image = preprocess_image(image)

    # Prediksi menggunakan model
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)  # Ambil indeks kelas dengan probabilitas tertinggi
    confidence = np.max(predictions)  # Ambil nilai confidence tertinggi

    # Menampilkan hasil prediksi
    st.write(f"Predicted Class: **{class_names[predicted_class]}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")

# Footer
st.write("Model trained using MobileNetV2 and deployed with Streamlit.")
