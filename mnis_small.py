from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Subset

transform = transforms.Compose([
    transforms.ToTensor(),  # wandelt Bild (PIL) in Tensor [1, 28, 28]
    transforms.Normalize((0.1307,), (0.3081,))  # Mittelwert und Std vom MNIST-Datenset
])

# Training-Set laden
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Test-Set laden
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, padding=1, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)  # Max-Pooling mit Kernel 2x2

        self.fc1 = nn.Linear(36, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 4)
        self.fc4 = nn.Linear(4, 10)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Durch die Convolutional Layers und das Pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Umformen des Tensors, um ihn in das Fully Connected Layer zu Ã¼bergeben
        x = x.view(x.size(0), -1)

        # Durch das Fully Connected Layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x) + x)
        x = F.relu(self.fc3(x) + x)
        x = self.fc4(x)
        return x


# Setting
batch_size = 64
epochs = 80
lr = 0.001

# DataLoader
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model init
model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss und Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)

# Train
for epoch in range(epochs):
    torch.set_grad_enabled(True)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  
        optimizer.step()

        running_loss += loss.item()

        # Berechne die Genauigkeit
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(
        f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(trainloader):.4f} - Accuracy: {100 * correct / total:.2f}%")

    # Evaluation
    torch.set_grad_enabled(False)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    correct_ = 0
    total_ = 0
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_ += labels.size(0)
        correct_ += (predicted == labels).sum().item()

    print(f"Test Loss: {running_loss / len(testloader):.4f} - Test Accuracy: {100 * correct_ / total_:.2f}%")

torch.save(model.state_dict(), "trained_model.pth")
