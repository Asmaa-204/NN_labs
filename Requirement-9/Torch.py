# No urgent need to understand all that's going here. It's a demo and you will implement from scratch and we will learn more about it later.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm  
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset



class ClassificationNN(nn.Module):
    def __init__(self):
        super(ClassificationNN, self).__init__()
        # Design a complex network because our task is complex
        self.model = nn.Sequential(
            nn.Linear(in_features=2, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        
        # Weight initalization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')



    def fit(self, x_data, y_data, epochs=250):
        x_data_tensor = torch.tensor(x_data, dtype=torch.float32)
        y_data_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)  # Reshape y_data to match the output shape
        
        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for binary classification
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # Convert numpy arrays to PyTorch DataLoader
        dataset = TensorDataset(x_data_tensor, y_data_tensor)
        train_loader = DataLoader(dataset, batch_size=1000, shuffle=True)

        # Training loop with tqdm over epochs
        with tqdm(range(epochs), desc="Epochs") as epoch_progress:
            for epoch in epoch_progress:
                running_loss = 0.0
                # Wrap your train_loader with tqdm
                for inputs, labels in train_loader:
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self(inputs)

                    # Calculate loss
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    # Print statistics
                    running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(train_loader.dataset)
                epoch_progress.set_postfix(loss=epoch_loss)  # Update progress bar with epoch loss


    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            return np.round(np.array(torch.sigmoid(self.forward(x))))
    
    

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_classification(model, x_data, μ, σ):
    x_data = x_data *σ + μ

    # Define a grid of points covering the input space
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                        np.arange(y_min, y_max, 1))

    # Make predictions on the grid points
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    grid_tensor_n = ((grid_tensor - μ)/σ).float()
    
    Z = model(grid_tensor_n).detach().numpy()
    # if less than 0.5 then 0 else 1
    Z[Z>0.5] = 1
    Z[Z<0.5] = 0
    Z = Z.reshape(xx.shape)


    # Plot the decision boundaries and data points
    plt.style.use('dark_background')
    plt.figure(figsize=(9, 6), dpi=150)
    plt.contourf(yy, xx, Z, alpha=0.5,  cmap=ListedColormap(('slategrey', 'red')))
    plt.title('Classifier Contours')
    plt.legend()
    plt.show()