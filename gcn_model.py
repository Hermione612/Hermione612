
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib


# 定义GCN模型
class GCN(nn.Module):
    def __init__(self, adjacency_matrix, input_size, num_hidden_units):
        super(GCN, self).__init__()
        self.adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.input_size = input_size
        self.num_hidden_units = num_hidden_units
        self.gcn_layer1 = nn.Linear(input_size, num_hidden_units)
        self.gcn_layer2 = nn.Linear(num_hidden_units, num_hidden_units)
        self.output_layer = nn.Linear(num_hidden_units, 8)

    def forward(self, x):
        x = torch.matmul(x, self.adjacency_matrix)
        x = self.gcn_layer1(x)
        x = F.relu(x)

        x = torch.matmul(x, self.adjacency_matrix)
        x = self.gcn_layer2(x)
        x = F.relu(x)

        return self.output_layer(x)


def main():
    # for beam
    distance_data = pd.read_csv('beam_distance.csv', index_col=0)
    strain_displacement_data = pd.read_csv('beam_data2.csv', header=None)
    strain_displacement_data = strain_displacement_data.T

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(strain_displacement_data.iloc[:, :8])
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(strain_displacement_data.iloc[:, 8:])
    joblib.dump(scaler_y, 'scaler_beam_y.joblib')
    joblib.dump(scaler_X, 'scaler_beam_x.joblib')

    threshold =6
    adjacency_matrix = np.where(distance_data.values < threshold, 1, 0)
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    degree_sqrt_inv = np.linalg.inv(np.sqrt(degree_matrix))
    normalized_adjacency = np.dot(np.dot(degree_sqrt_inv, adjacency_matrix), degree_sqrt_inv)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=10)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# for plate_tri
    distance_data1 = pd.read_csv('plate_tri_distance1.csv', index_col=0)
    strain_displacement_data1 = pd.read_csv('plate_tri_data12.csv', header=None)
    strain_displacement_data1 = strain_displacement_data1.T

    scaler_X1 = StandardScaler()
    X_scaled1 = scaler_X1.fit_transform(strain_displacement_data1.iloc[:, :8])
    scaler_y1 = StandardScaler()
    y_scaled1 = scaler_y1.fit_transform(strain_displacement_data1.iloc[:, 8:])
    joblib.dump(scaler_y1, 'scaler_platetri_y.joblib')
    joblib.dump(scaler_X1, 'scaler_platetri_x.joblib')

    threshold1 = 6
    adjacency_matrix1 = np.where(distance_data1.values < threshold1, 1, 0)
    degree_matrix1 = np.diag(np.sum(adjacency_matrix1, axis=1))
    degree_sqrt_inv1 = np.linalg.inv(np.sqrt(degree_matrix1))
    normalized_adjacency1 = np.dot(np.dot(degree_sqrt_inv1, adjacency_matrix1), degree_sqrt_inv1)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled1, y_scaled1, test_size=0.1, random_state=20)

    X_train_tensor1 = torch.tensor(X_train1, dtype=torch.float32)
    y_train_tensor1 = torch.tensor(y_train1, dtype=torch.float32)
    X_test_tensor1 = torch.tensor(X_test1, dtype=torch.float32)
    y_test_tensor1 = torch.tensor(y_test1, dtype=torch.float32)

    # for plate_yi
    distance_data2 = pd.read_csv('plate_yi_distance1.csv', index_col=0)
    strain_displacement_data2 = pd.read_csv('plate_yi_data12.csv', header=None)
    strain_displacement_data2 = strain_displacement_data2.T

    scaler_X2 = StandardScaler()
    X_scaled2 = scaler_X2.fit_transform(strain_displacement_data2.iloc[:, :8])
    scaler_y2 = StandardScaler()
    y_scaled2 = scaler_y2.fit_transform(strain_displacement_data2.iloc[:, 8:])
    joblib.dump(scaler_y2, 'scaler_plateyi_y.joblib')
    joblib.dump(scaler_X2, 'scaler_plateyi_x.joblib')

    threshold2 = 6
    adjacency_matrix2 = np.where(distance_data2.values < threshold2, 1, 0)
    degree_matrix2 = np.diag(np.sum(adjacency_matrix2, axis=1))
    degree_sqrt_inv2 = np.linalg.inv(np.sqrt(degree_matrix2))
    normalized_adjacency2 = np.dot(np.dot(degree_sqrt_inv2, adjacency_matrix2), degree_sqrt_inv2)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_scaled2, y_scaled2, test_size=0.1, random_state=20)

    X_train_tensor2 = torch.tensor(X_train2, dtype=torch.float32)
    y_train_tensor2 = torch.tensor(y_train2, dtype=torch.float32)
    X_test_tensor2 = torch.tensor(X_test2, dtype=torch.float32)
    y_test_tensor12= torch.tensor(y_test2, dtype=torch.float32)

    gcn_model = GCN(normalized_adjacency, input_size=8, num_hidden_units=8)
    gcn_model1 = GCN(normalized_adjacency1, input_size=8, num_hidden_units=8)
    gcn_model2 = GCN(normalized_adjacency2, input_size=8, num_hidden_units=8)

    optimizer = optim.Adam(gcn_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    optimizer1 = optim.Adam(gcn_model1.parameters(), lr=0.01)
    criterion1 = nn.MSELoss()

    optimizer2 = optim.Adam(gcn_model2.parameters(), lr=0.01)
    criterion2 = nn.MSELoss()

    # 训练模型
    train_losses = []
    train_losses1 = []
    train_losses2 = []
    print("training for beam")
    for epoch in range(400):
        # 训练,26 nodes
        gcn_model.train()
        optimizer.zero_grad()
        outputs = gcn_model(X_train_tensor)
        loss = criterion(outputs,
                         y_train_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/1000], Training Loss: {loss.item():.4f}')
    print("training for plate triangle")
    for epoch in range(400):
        gcn_model1.train()
        optimizer1.zero_grad()
        outputs1 = gcn_model1(X_train_tensor1)
        loss1 = criterion1(outputs1, y_train_tensor1)
        loss1.backward()
        optimizer1.step()
        train_losses1.append(loss1.item())
        if epoch % 10 == 0:
               print(f'Epoch [{epoch + 1}/1000], Training Loss: {loss1.item():.4f}')
    print("training for plate triangle")
    for epoch in range(400):
        gcn_model2.train()
        optimizer2.zero_grad()
        outputs2 = gcn_model2(X_train_tensor2)
        loss2 = criterion1(outputs2, y_train_tensor2)
        loss2.backward()
        optimizer2.step()
        train_losses2.append(loss2.item())
        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/1000], Training Loss: {loss2.item():.4f}')

    torch.save(gcn_model.state_dict(), 'model_beam.pth')
    torch.save(gcn_model1.state_dict(), 'model_plateyi.pth')
    torch.save(gcn_model1.state_dict(), 'model_platetri.pth')

