import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, 3, padding=1)

        self.pool = nn.MaxPool3d(2)

        self.fc1 = nn.Linear(16 * 16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x



def load_trained_model(path="model/model.pt"):
    model = Simple3DCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
