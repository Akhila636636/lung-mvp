import torch
import torch.nn as nn
import torch.optim as optim
from model import Simple3DCNN

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Simple3DCNN().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---- DUMMY DATA ----
    x = torch.randn(8, 1, 64, 64, 64).to(device)
    y = torch.randint(0, 2, (8, 1)).float().to(device)

    for epoch in range(3):
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "model.pt")
    print("Model saved as model.pt")


if __name__ == "__main__":
    train()
