import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt

# Step 1: Train and save a CNN modelp
def train_and_save_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Added dropout to prevent overfitting
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))  # Increased epochs
    model.save("digit_recognizer_model.keras")

# Check if model exists, otherwise train and save
if not os.path.exists("digit_recognizer_model.keras"):
    train_and_save_model()

# Step 2: GUI for drawing digits
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.canvas = tk.Canvas(self, width=200, height=200, bg='white')
        self.canvas.pack()
        self.button = tk.Button(self, text="Recognize", command=self.predict_digit)
        self.button.pack()
        self.clear_button = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        self.label = tk.Label(self, text="", font=("Helvetica", 24))
        self.label.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new("L", (200, 200), 255)
        self.draw_img = ImageDraw.Draw(self.image)
        self.model = load_model("digit_recognizer_model.keras", compile=False)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw_img.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_img.rectangle([0, 0, 200, 200], fill=255)
        self.label.config(text="")

    def predict_digit(self):
        # Convert the canvas image to a PIL image
        img = self.image.copy()

        # Resize the image to 28x28 pixels
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # Ensure the image is in grayscale mode
        if img.mode != "L":
            img = img.convert("L")

        # Invert the image (white background to black background)
        img = ImageOps.invert(img)

        # Normalize pixel values to the range [0, 1]
        img = np.array(img).astype(np.float32) / 255.0

        # Reshape the image to match the input shape of the model
        img = img.reshape(1, 28, 28, 1)

        # Debug: Visualize the preprocessed image
        plt.imshow(img[0, :, :, 0], cmap='gray')
        plt.title("Preprocessed Input Image")
        plt.show()

        # Predict the digit
        prediction = self.model.predict(img)
        digit = np.argmax(prediction)

        # Display the predicted digit
        self.label.config(text=f"Prediction: {digit}")

# Run the GUI
app = App()
app.mainloop()
