import torch, torch.nn as nn, torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from dataset import VideoFrameDataset

class CNNLSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=10):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = self.feature_extractor(x.view(B*T, C, H, W)).view(B, T, -1)
        out, _ = self.lstm(feats)
        return self.fc(out[:, -1, :])

def main():
    transform = transforms.Compose([transforms.Resize((112,112)), transforms.ToTensor()])
    dataset = VideoFrameDataset("data/ucf101_frames", transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = CNNLSTM(num_classes=len(dataset.classes)).cuda()
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(2):
        for images, labels in loader:
            images = torch.stack(images, dim=1).cuda()
            labels = torch.tensor(labels).cuda()
            outputs, loss = model(images), criterion(model(images), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"Epoch {epoch} Loss {loss.item():.4f}")

if __name__ == "__main__":
    main()
