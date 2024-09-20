import tkinter as tk
from tkinter import messagebox, filedialog
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
import torch
from gcn_model import GCN


class PredictionApp(tk.Tk):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.title("GCN Prediction App")
        self.geometry("500x300")
        self.init_main_ui()

    def init_main_ui(self):
        tk.Button(self, text="Beam structure", command=lambda: self.open_prediction_ui('Beam structure')).pack(pady=10)
        tk.Button(self, text="Square plate structure", command=lambda: self.open_prediction_ui('Square plate structure')).pack(pady=10)
        tk.Button(self, text="Triangle plate structure", command=lambda: self.open_prediction_ui('Triangle plate structure')).pack(pady=10)

    def open_prediction_ui(self, structure_type):
        # Close the main window and open the prediction interface
        self.withdraw()
        prediction_window = PredictionWindow(self, 8, self.models[structure_type])
        prediction_window.grab_set()


class PredictionWindow(tk.Toplevel):
    def __init__(self, parent, num_nodes, model_info):
        super().__init__(parent)
        self.model = model_info['model']
        self.scaler_y = model_info['scalery']
        self.scaler_x = model_info['scalerx']
        self.num_nodes = num_nodes
        self.parent = parent
        self.strain_values = [tk.StringVar() for _ in range(num_nodes)]
        self.result_vars = [tk.StringVar() for _ in range(num_nodes)]

        self.title(f"Prediction for {model_info['name']} Displacement")

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        tk.Label(self, text="Enter strains individually or as an array:").pack()
        tk.Frame(self, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=5, pady=5)
        entry_frame = tk.Frame(self)
        entry_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Assuming num_nodes is 8 and they are displayed in two rows of 4 each.
        rows = 2
        cols = 4
        entries = [None] * self.num_nodes

        for i in range(self.num_nodes):
            row = tk.Frame(entry_frame)
            tk.Label(row, text=f"Node {i + 1} Strain:").pack(side=tk.LEFT)
            entries[i] = tk.Entry(row, textvariable=self.strain_values[i])
            entries[i].pack(side=tk.RIGHT)
            row.grid(row=i // cols, column=i % cols)

        self.array_entry = tk.Entry(self)
        self.array_entry.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Button(self, text="Predict", command=self.predict_displacement).pack(pady=10)

        result_frame = tk.Frame(self)
        result_frame.pack(fill=tk.BOTH, expand=True)
        for i in range(self.num_nodes):
            tk.Label(result_frame, textvariable=self.result_vars[i]).grid(row=i // cols, column=i % cols)

        tk.Button(self, text="Save Results", command=self.save_results).pack(pady=10)
        tk.Button(self, text="Back", command=self.go_back).pack(pady=10)

    def predict_displacement(self):
        try:
            if self.array_entry.get():
                strains = np.fromstring(self.array_entry.get(), sep='\t', dtype=float)
            else:
                strains = np.array([float(v.get()) for v in self.strain_values], dtype=float)
            if len(strains) != self.num_nodes:
                print(len(strains))
                raise ValueError("Input array does not match node count")

            # Preparing input tensor
            strains = self.scaler_x.transform(np.array([strains]))
            strains_tensor = torch.from_numpy(strains).float()

            with torch.no_grad():
                self.model.eval()
                predicted_displacement_scaled = self.model(strains_tensor).numpy()
                predicted_displacement = self.scaler_y.inverse_transform(predicted_displacement_scaled)

            # Updating the GUI with unscaled results
            for i, value in enumerate(predicted_displacement.flatten()):
                self.result_vars[i].set(f"Node {i + 1} Displacement: {value:.4f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_results(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if file_path:
            data = {f"Node {i + 1}": [v.get()] for i, v in enumerate(self.result_vars)}
            df = pd.DataFrame(data)
            df.to_excel(file_path, index=False)
            messagebox.showinfo("Success", "Results saved successfully.")

    def go_back(self):
        self.destroy()  # 销毁当前窗口
        self.parent.deiconify()  # 重新显示父窗口

    def on_close(self):
        self.go_back()
        self.destroy()
        self.parent.deiconify()


if __name__ == "__main__":
    scaler_beam_y = joblib.load('_internal/.parameters/scaler_beam_y.joblib')
    scaler_beam_x = joblib.load('_internal/.parameters/scaler_beam_x.joblib')
    scaler_platetri_y = joblib.load('_internal/.parameters/scaler_platetri_y.joblib')
    scaler_platetri_x = joblib.load('_internal/.parameters/scaler_platetri_x.joblib')
    scaler_plateyi_y = joblib.load('_internal/.parameters/scaler_plateyi_y.joblib')
    scaler_plateyi_x = joblib.load('_internal/.parameters/scaler_plateyi_x.joblib')

    distance_data = pd.read_csv('_internal/.distances/beam_distance.csv', index_col=0)
    threshold = 6
    adjacency_matrix = np.where(distance_data.values < threshold, 1, 0)
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    degree_sqrt_inv = np.linalg.inv(np.sqrt(degree_matrix))
    normalized_adjacency = np.dot(np.dot(degree_sqrt_inv, adjacency_matrix), degree_sqrt_inv)

    distance_data1 = pd.read_csv('_internal/.distances/plate_tri_distance1.csv', index_col=0)
    threshold1 = 6
    adjacency_matrix1 = np.where(distance_data1.values < threshold1, 1, 0)
    degree_matrix1 = np.diag(np.sum(adjacency_matrix1, axis=1))
    degree_sqrt_inv1 = np.linalg.inv(np.sqrt(degree_matrix1))
    normalized_adjacency1 = np.dot(np.dot(degree_sqrt_inv1, adjacency_matrix1), degree_sqrt_inv1)

    distance_data2 = pd.read_csv('_internal/.distances/plate_yi_distance1.csv', index_col=0)
    threshold2 = 6
    adjacency_matrix2 = np.where(distance_data2.values < threshold2, 1, 0)
    degree_matrix2 = np.diag(np.sum(adjacency_matrix2, axis=1))
    degree_sqrt_inv2 = np.linalg.inv(np.sqrt(degree_matrix2))
    normalized_adjacency2 = np.dot(np.dot(degree_sqrt_inv2, adjacency_matrix2), degree_sqrt_inv2)

    # 创建模型实例并加载状态字典
    model_beam = GCN(normalized_adjacency, input_size=8, num_hidden_units=8)
    model_beam.load_state_dict(torch.load('_internal/.weights/model_beam.pth'))

    model_platetri = GCN(normalized_adjacency1, input_size=8, num_hidden_units=8)
    model_platetri.load_state_dict(torch.load('_internal/.weights/model_platetri.pth'))

    model_plateyi = GCN(normalized_adjacency2, input_size=8, num_hidden_units=8)
    model_plateyi.load_state_dict(torch.load('_internal/.weights/model_plateyi.pth'))


    models = {
        'Beam structure': {
            'model': model_beam,
            'scalery': scaler_beam_y,
            'scalerx': scaler_beam_x,
            'name': 'Beam structure'
        },
        'Square plate structure': {
            'model': model_platetri,
            'scalery': scaler_platetri_y,
            'scalerx': scaler_platetri_x,
            'name': 'Square plate structure'
        },
        'Triangle plate structure': {
            'model': model_plateyi,
            'scalery': scaler_plateyi_y,
            'scalerx': scaler_plateyi_x,
            'name': 'Triangle plate structure'
        }
    }

    app = PredictionApp(models)
    app.mainloop()