import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tkinter import *
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model('my_model.h5')

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = Canvas(root, width=280, height=280, bg='white')
        self.canvas.grid(row=0, column=0, pady=20, padx=20)

        self.clear_button = Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0, pady=10)

        self.predict_button = Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.grid(row=2, column=0, pady=10)

        self.image = Image.new("L", (280, 280), 0)  # Черный фон
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', width=4)
        self.draw.line([x1, y1, x2, y2], fill=255, width=4)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)

        self.show_prediction(predicted_label)

    def show_prediction(self, predicted_label):
        result_window = Toplevel(self.root)
        result_window.title("Prediction Result")

        label = Label(result_window, text=f"Predicted Digit: {predicted_label}", font=("Arial", 20))
        label.pack(padx=20, pady=20)

        button_close = Button(result_window, text="Close", command=result_window.destroy)
        button_close.pack(pady=10)


root = Tk()
app = DigitRecognizerApp(root)
root.mainloop()
