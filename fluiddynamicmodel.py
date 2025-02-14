import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

class FullyFluidLayer(nn.Module):
    def __init__(self, input_size, output_size, diffusion=0.01, viscosity=0.001, time_steps=50):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.diffusion = diffusion
        self.viscosity = viscosity
        self.time_steps = time_steps

        self.pressure_weights = nn.Parameter(torch.randn(input_size, output_size))
        self.velocity_x = nn.Parameter(torch.zeros(1, output_size))  # x-velocity
        self.velocity_y = nn.Parameter(torch.zeros(1, output_size))  # y-velocity

    def forward(self, x):
        pressure = x @ self.pressure_weights
        batch_size = x.shape[0]
        output_states = []

        for i in range(batch_size):
            # Process each item in the batch independently
            state = pressure[i:i+1, :].clone() #Crucially, keep it as a 2D tensor.

            for _ in range(self.time_steps):
                laplacian = (torch.roll(state, shifts=1, dims=1) +
                             torch.roll(state, shifts=-1, dims=1) - 2 * state)

                # Update velocities (using .data for in-place modification)
                self.velocity_x.data = self.velocity_x.data - self.diffusion * (torch.roll(state, shifts=-1, dims=1) - state)
                self.velocity_y.data = self.velocity_y.data - self.diffusion * (torch.roll(state, shifts=1, dims=1) - state)

                # Clamp velocities
                self.velocity_x.data = torch.clamp(self.velocity_x.data, -1, 1)
                self.velocity_y.data = torch.clamp(self.velocity_y.data, -1, 1)

                advection_x = self.velocity_x * (torch.roll(state, shifts=-1, dims=1) - state)
                advection_y = self.velocity_y * (torch.roll(state, shifts=1, dims=1) - state)

                state = state + self.diffusion * laplacian - self.viscosity * state - advection_x - advection_y

                # Clamp state
                state = torch.clamp(state, -10, 10)
            output_states.append(state)
        # Stack to restore batch dimension
        return torch.cat(output_states, dim=0)

class FullyFluidNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_steps=50):
        super().__init__()
        self.layer1 = FullyFluidLayer(input_size, hidden_size, time_steps=time_steps)
        self.layer2 = FullyFluidLayer(hidden_size, hidden_size, time_steps=time_steps)
        self.output_layer = FullyFluidLayer(hidden_size, output_size, time_steps=time_steps)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=-1)

# --- Data Loading and Preprocessing ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# --- Model, Optimizer, and Training ---
input_size = 28 * 28
hidden_size = 256
output_size = 10
time_steps = 50

model = FullyFluidNetwork(input_size, hidden_size, output_size, time_steps)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 28 * 28)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy


# --- Main Training and Evaluation Loop ---
num_epochs = 10
best_accuracy = 0
for epoch in range(1, num_epochs + 1):
    train(model, train_loader, optimizer, epoch)
    accuracy = test(model, test_loader)
    if accuracy > best_accuracy:
      best_accuracy = accuracy

print(f"Best accuracy achieved: {best_accuracy:.2f}%")
