import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_frames=16):
        self.samples, self.classes = [], sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform, self.max_frames = transform, max_frames

        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for video in os.listdir(class_dir):
                frame_dir = os.path.join(class_dir, video)
                frames = sorted(os.listdir(frame_dir))
                frame_paths = [os.path.join(frame_dir, f) for f in frames]
                self.samples.append((frame_paths, self.class_to_idx[cls]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        images = [Image.open(fp).convert("RGB") for fp in frame_paths[:self.max_frames]]
        if self.transform: images = [self.transform(img) for img in images]
        return images, label
