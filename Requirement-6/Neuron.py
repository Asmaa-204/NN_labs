import torch
import numpy as np

# 1. Define hypothesis f(x;θ)
def σ(z):
    return 1 / (1 + torch.exp(-z))

def f(u_data, weights):                 # m row, n+1 columns 
    return σ(u_data @ weights)

# 2. Define loss function
def J(y_pred, y_true):
    ϵ = 1e-7                                    # Small value to avoid log(0)
    ŷ, y = y_pred, y_true
    return -torch.mean(y * torch.log(ŷ + ϵ) + (1 - y) * torch.log(1 - ŷ + ϵ))

# 3. Optimize the loss over θ for training and plug back in hypothesis for prediction
class LogisticRegression:
    def __init__(self):
        self.weights = None         # w0, w1, ..., wn

    def fit(self, x_train, y_train, α=0.01, num_epochs=100, verbose=False):
        # Handle Numpy arrays
        u_train = torch.tensor(np.c_[np.ones(len(x_train)), x_train], dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        
        # Initialize weights
        n = u_train.shape[1]
        self.weights = torch.zeros((n,), dtype=torch.float32, requires_grad=True)       # w0, w1, w2, ..., wn
        
        for epoch in range(num_epochs):                     # i.e., number of iterations
            # 1. Forward pass to compute loss
            y_pred = f(u_train, self.weights)
            loss = J(y_pred, y_train)
            
            # 2. Backward pass to compute მJⳆმw
            loss.backward()
            
            # Update weights
            with torch.no_grad():
                self.weights -=  α * self.weights.grad
                
                # Reset gradients
                self.weights.grad.zero_()
            
            if verbose and (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
    def predict(self, x_val):
        # Handle Numpy arrays
        x_val = torch.tensor(np.c_[np.ones(len(x_val)), x_val], dtype=torch.float32)
        
        y_pred_probs = f(x_val, self.weights)
        return np.array((y_pred_probs > 0.5).float())             # Convert probabilities to binary classes