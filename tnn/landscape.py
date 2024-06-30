import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Dummy dataset
X = torch.randn(100, 1, 28, 28)
y = torch.randint(0, 2, (100,)).long()


# Simple ResNet-like model without skip connections
class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.bn1(f.relu(self.conv1(x)))
        x = self.bn2(f.relu(self.conv2(x)))
        x = self.bn3(f.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleResNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Choose two directions in the parameter space
w = list(model.parameters())[0].detach().numpy()
v = list(model.parameters())[1].detach().numpy()

# Create a grid of points
grid_size = 20
w_range = np.linspace(w[0, 0, 0, 0] - 1, w[0, 0, 0, 0] + 1, grid_size)
v_range = np.linspace(v[0] - 1, v[0] + 1, grid_size)

W, V = np.meshgrid(w_range, v_range)
loss_grid = np.zeros((grid_size, grid_size))

# Compute the loss for each point in the grid
for i in range(grid_size):
    for j in range(grid_size):
        w_new = torch.tensor(w)
        v_new = torch.tensor(v)
        w_new[0, 0, 0, 0] = W[i, j]
        v_new[0] = V[i, j]
        with torch.no_grad():
            model.conv1.weight = nn.Parameter(w_new.float())
            model.conv1.bias = nn.Parameter(v_new.float())
        outputs = model(X)
        loss = criterion(outputs, y)
        loss_grid[i, j] = loss.item()

# Plot the 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(W, V, loss_grid, cmap="viridis")

ax.set_xlabel("Weight[0, 0, 0, 0]")
ax.set_ylabel("Bias[0]")
ax.set_zlabel("Loss")
ax.set_title("3D Loss Landscape")

plt.show()
