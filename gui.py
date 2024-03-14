import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image
from digit_classifier import DigitClassifier

class DigitClassApp:
    def __init__(self, root):
        self.root = root
        self.canvas_width = 600
        self.canvas_height = 600
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white", bd=3, relief=tk.SUNKEN)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.setup_tools()
        self.setup_events()
        self.prev_x = None
        self.prev_y = None
        self.classifier = DigitClassifier('model.dat')

    def setup_tools(self):
        self.selected_tool = "pen"
        self.selected_color = "black"
        self.selected_size = 35

        self.tool_frame = ttk.LabelFrame(self.root, text='Predictions')
        self.tool_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.Y)
        
        self.probs = np.zeros(10)
        self.prediction = None
        
        self.prediction_label = ttk.Label(self.tool_frame, text=f"Prediction: {self.prediction}", font='bold')
        self.prediction_label.pack(side=tk.TOP, padx=5, pady=20)
        
        self.label0 = ttk.Label(self.tool_frame, text=f"0: {round(100 * self.probs[0], 2)}%")
        self.label1 = ttk.Label(self.tool_frame, text=f"1: {round(100 * self.probs[1], 2)}%")
        self.label2 = ttk.Label(self.tool_frame, text=f"2: {round(100 * self.probs[2], 2)}%")
        self.label3 = ttk.Label(self.tool_frame, text=f"3: {round(100 * self.probs[3], 2)}%")
        self.label4 = ttk.Label(self.tool_frame, text=f"4: {round(100 * self.probs[4], 2)}%")
        self.label5 = ttk.Label(self.tool_frame, text=f"5: {round(100 * self.probs[5], 2)}%")
        self.label6 = ttk.Label(self.tool_frame, text=f"6: {round(100 * self.probs[6], 2)}%")
        self.label7 = ttk.Label(self.tool_frame, text=f"7: {round(100 * self.probs[7], 2)}%")
        self.label8 = ttk.Label(self.tool_frame, text=f"8: {round(100 * self.probs[8], 2)}%")
        self.label9 = ttk.Label(self.tool_frame, text=f"9: {round(100 * self.probs[9], 2)}%")
        
        self.label0.pack(side=tk.TOP, padx=5, pady=10)
        self.label1.pack(side=tk.TOP, padx=5, pady=10)
        self.label2.pack(side=tk.TOP, padx=5, pady=10)
        self.label3.pack(side=tk.TOP, padx=5, pady=10)
        self.label4.pack(side=tk.TOP, padx=5, pady=10)
        self.label5.pack(side=tk.TOP, padx=5, pady=10)
        self.label6.pack(side=tk.TOP, padx=5, pady=10)
        self.label7.pack(side=tk.TOP, padx=5, pady=10)
        self.label8.pack(side=tk.TOP, padx=5, pady=10)
        self.label9.pack(side=tk.TOP, padx=5, pady=10)

        self.clear_button = ttk.Button(self.tool_frame, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack(side=tk.BOTTOM, padx=5, pady=5)
        
        self.predict_button = ttk.Button(self.tool_frame, text="Predict", command=self.evaluate)
        self.predict_button.pack(side=tk.BOTTOM, padx=5, pady=5)

    def setup_events(self):
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.release)

    def draw(self, event):
        if self.selected_tool == "pen":
            if self.prev_x is not None and self.prev_y is not None:
                self.canvas.create_oval(self.prev_x, self.prev_y, event.x, event.y, fill=self.selected_color, width=self.selected_size)
            self.prev_x = event.x
            self.prev_y = event.y

    def release(self, event):
        self.prev_x = None
        self.prev_y = None

    def clear_canvas(self):
        self.canvas.delete("all")

    def evaluate(self):
        filename = 'canvas.eps'
        self.canvas.postscript(file = filename) 
        img = np.asarray(Image.open(filename).resize((28, 28)).convert('L'))
        self.probs = self.classifier.predict_probabilities(img)
        self.prediction = self.classifier.predict(img)
        self.update_prediction_labels()
    
    def update_prediction_labels(self):
        self.prediction_label['text'] = f"Prediction: {self.prediction}"
        self.label0['text'] = f"0: {round(100 * self.probs[0], 2)}%"
        self.label1['text'] = f"1: {round(100 * self.probs[1], 2)}%"
        self.label2['text'] = f"2: {round(100 * self.probs[2], 2)}%"
        self.label3['text'] = f"3: {round(100 * self.probs[3], 2)}%"
        self.label4['text'] = f"4: {round(100 * self.probs[4], 2)}%"
        self.label5['text'] = f"5: {round(100 * self.probs[5], 2)}%"
        self.label6['text'] = f"6: {round(100 * self.probs[6], 2)}%"
        self.label7['text'] = f"7: {round(100 * self.probs[7], 2)}%"
        self.label8['text'] = f"8: {round(100 * self.probs[8], 2)}%"
        self.label9['text'] = f"9: {round(100 * self.probs[9], 2)}%"

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Digit Classifier")
    app = DigitClassApp(root)
    root.mainloop()