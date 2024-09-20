import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import pandas as pd
import torch

class PredictionApp(tk.Tk):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.title("GCN Prediction App")
        self.init_main_ui()

    def init_main_ui(self):
        tk.Button(self, text="26 Node Prediction", command=lambda: self.open_prediction_ui(26)).pack(pady=10)
        tk.Button(self, text="45 Node Prediction", command=lambda: self.open_prediction_ui(45)).pack(pady=10)

    def open_prediction_ui(self, num_nodes):
        # Close the main window and open the prediction interface
        self.withdraw()
        prediction_window = PredictionWindow(self, num_nodes, self.models[num_nodes])
        prediction_window.grab_set()

class PredictionWindow(tk.Toplevel):
    def __init__(self, parent, num_nodes, model):
        super().__init__(parent)
        self.model = model
        self.num_nodes = num_nodes
        self.parent = parent
        self.strain_values = [tk.StringVar() for _ in range(num_nodes)]
        self.result_vars = [tk.StringVar() for _ in range(num_nodes)]
        self.title(f"{num_nodes} Node Prediction")

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        tk.Label(self, text="Enter strains individually or as an array:").pack()
        tk.Frame(self, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=5, pady=5)

        for i in range(self.num_nodes):
            row = tk.Frame(self)
            tk.Label(row, text=f"Node {i+1} Strain:").pack(side=tk.LEFT)
            tk.Entry(row, textvariable=self.strain_values[i]).pack(side=tk.RIGHT)
            row.pack()

        self.array_entry = tk.Entry(self)
        self.array_entry.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Button(self, text="Predict", command=self.predict_displacement).pack(pady=10)

        result_frame = tk.Frame(self)
        result_frame.pack(fill=tk.BOTH, expand=True)
        for i in range(self.num_nodes):
            tk.Label(result_frame, textvariable=self.result_vars[i]).grid(row=i//5, column=i%5)

        tk.Button(self, text="Save Results", command=self.save_results).pack(pady=10)

    def predict_displacement(self):
        try:
            if self.array_entry.get():
                strains = np.fromstring(self.array_entry.get(), sep=',', dtype=float)
            else:
                strains = np.array([float(v.get()) for v in self.strain_values], dtype=float)
            if len(strains) != self.num_nodes:
                raise ValueError("Input array does not match node count")
            with torch.no_grad():
                self.model.eval()
                prediction = self.model(torch.tensor([strains], dtype=torch.float32)).numpy().flatten()
            for i, value in enumerate(prediction):
                self.result_vars[i].set(f"Node {i+1} Displacement: {value:.4f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_results(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if file_path:
            data = {f"Node {i+1}": [v.get()] for i, v in enumerate(self.result_vars)}
            df = pd.DataFrame(data)
            df.to_excel(file_path, index=False)
            messagebox.showinfo("Success", "Results saved successfully.")

    def on_close(self):
        self.destroy()
        self.parent.deiconify()

