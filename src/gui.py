import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('tomato_leaf_disease_model.h5')

# Class names
classes = ['Bacterial_spot', 'Early_blight', 'healthy', 'Late_blight', 'Leaf_Mold', 'powdery_mildew', 'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_mosaic_virus', 'Tomato_Yellow_Leaf_Curl_Virus']

def classify_image(image_path):
    image = plt.imread(image_path)
    image = np.resize(image, (224, 224, 3))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return classes[predicted_class]

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and display the image
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img_tk = ImageTk.PhotoImage(img)
        panel.configure(image=img_tk)
        panel.image = img_tk

        # Classify the image
        result = classify_image(file_path)
        result_label.config(text=f"Prediction: {result}")

# Create the Tkinter window
root = tk.Tk()
root.title("Tomato Leaf Disease Classification")
root.geometry("600x600")

# Create and place the widgets
upload_button = ttk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

panel = tk.Label(root)
panel.pack(pady=20)

result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 16))
result_label.pack(pady=20)

root.mainloop()
