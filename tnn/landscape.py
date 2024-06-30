import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Dummy dataset
X = torch.randn(100, 2)
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

# Store parameter trajectories
w_trajectory = []
v_trajectory = []

# Train the model briefly
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    # Record parameters
    w = model.fc.weight.data.clone().numpy()
    v = model.fc.bias.data.clone().numpy()
    w_trajectory.append(w[0, 0])
    v_trajectory.append(v[0])

# Choose two directions in the parameter space
w_init = w_trajectory[0]
v_init = v_trajectory[0]

# Create a grid of points
grid_size = 50
w_range = np.linspace(min(w_trajectory) - 1, max(w_trajectory) + 1, grid_size)
v_range = np.linspace(min(v_trajectory) - 1, max(v_trajectory) + 1, grid_size)

W, V = np.meshgrid(w_range, v_range)
loss_grid = np.zeros((grid_size, grid_size))

# Compute the loss for each point in the grid
for i in range(grid_size):
    for j in range(grid_size):
        w_new = torch.tensor([[W[i, j], w_init], [w_init, w_init]], requires_grad=True)
        v_new = torch.tensor([V[i, j], v_init], requires_grad=True)
        with torch.no_grad():
            model.fc.weight = nn.Parameter(w_new)
            model.fc.bias = nn.Parameter(v_new)
        outputs = model(X)
        loss = criterion(outputs, y)
        loss_grid[i, j] = loss.item()

# Plot the contour and trajectory
plt.contour(W, V, loss_grid, levels=20)
plt.plot(w_trajectory, v_trajectory, marker="o", color="red")
plt.xlabel("Weight[0, 0]")
plt.ylabel("Bias[0]")
plt.title("Loss Landscape Contour with Trajectory")
plt.show()
