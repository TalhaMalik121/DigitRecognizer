
# 🧠 Handwritten Digit Recognizer using CNN and Tkinter

This project is a **handwritten digit recognition tool** built with **TensorFlow/Keras** and **Tkinter**. It allows users to draw digits (0–9) on a canvas and uses a **Convolutional Neural Network (CNN)** trained on the MNIST dataset to predict the drawn digit.

---

## 📷 Demo

*Draw → Predict → Clear*

---

## 🚀 Features

- Digit drawing canvas (GUI using Tkinter)
- Real-time digit recognition
- Trained CNN model on MNIST dataset
- Option to retrain model if not present
- Debug view of the preprocessed image

---

## 🧩 Technologies Used

- Python
- TensorFlow / Keras
- Tkinter (for GUI)
- PIL (Python Imaging Library)
- Matplotlib (for debug visualization)
- NumPy

---

## 🛠️ How to Run

1. **Clone the repo** or copy the script.
2. Install the required dependencies:
   ```bash
   pip install tkinter tensorflow pillow matplotlib numpy
   ```
3. **Run the script**:
   ```bash
   python main.py
   ```

---

## 📁 File Structure

```
digit_recognizer/
│
├── main.py       # Main script
├── digit_recognizer_model.keras (auto-generated after training)
└── README.md
```

---

## 🧠 Model Details

The CNN model architecture:
- Conv2D (32 filters, 3x3, ReLU)
- MaxPooling2D
- Conv2D (64 filters, 3x3, ReLU)
- MaxPooling2D
- Flatten
- Dense (128 units, ReLU)
- Dropout (0.5)
- Dense (10 units, Softmax)

Trained on MNIST for 10 epochs.

---

## 📌 How It Works

1. User draws a digit (0–9) using the mouse.
2. Image is captured and resized to 28x28 pixels.
3. Image is converted to grayscale and inverted.
4. CNN model predicts the digit.
5. Prediction is displayed on screen.

---

## 📃 License

This project is open source and free to use for educational or non-commercial purposes.
