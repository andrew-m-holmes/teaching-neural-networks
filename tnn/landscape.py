import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Dummy dataset
X = torch.randn(100, 2).float()
y = torch.randint(0, 2, (100,)).long()


# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)


model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model briefly
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Choose two directions in the parameter space
w = list(model.parameters())[0].detach().numpy()
v = list(model.parameters())[1].detach().numpy()

# Create a grid of points
grid_size = 50
w_range = np.linspace(w[0, 0] - 1, w[0, 0] + 1, grid_size)
v_range = np.linspace(v[0] - 1, v[0] + 1, grid_size)

W, V = np.meshgrid(w_range, v_range)
loss_grid = np.zeros((grid_size, grid_size))

# Compute the loss for each point in the grid
for i in range(grid_size):
    for j in range(grid_size):
        w_new = torch.tensor(
            [[W[i, j], w[0, 1]], [w[1, 0], w[1, 1]]], requires_grad=True
        )
        v_new = torch.tensor([V[i, j], v[1]], requires_grad=True)
        with torch.no_grad():
            model.fc.weight = nn.Parameter(w_new.float())
            model.fc.bias = nn.Parameter(v_new.float())
        outputs = model(X)
        loss = criterion(outputs, y)
        loss_grid[i, j] = loss.item()

# Plot the contour
plt.contour(W, V, loss_grid, levels=20)
plt.xlabel("Direction 1")
plt.ylabel("Direction 2")
plt.title("Loss Landscape Contour")
plt.show()
