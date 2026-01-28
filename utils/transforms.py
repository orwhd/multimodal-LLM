from __future__ import annotations

import torch
import torchvision.transforms.functional as F
from torchvision import transforms

#for anyres transform
class AnyResTransform:
    def __init__(self, size: int, train: bool = False):
        self.size = size
        self.train = train
        
    def __call__(self, image):
        if self.train:
            if torch.rand(1) < 0.5:
                image = F.hflip(image)
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            image = jitter(image)

        global_view = transforms.Resize((self.size, self.size))(image)
        
        large_view = transforms.Resize((self.size * 2, self.size * 2))(image)
        tl = F.crop(large_view, 0, 0, self.size, self.size)
        tr = F.crop(large_view, 0, self.size, self.size, self.size)
        bl = F.crop(large_view, self.size, 0, self.size, self.size)
        br = F.crop(large_view, self.size, self.size, self.size, self.size)
        
        patches = [global_view, tl, tr, bl, br]
        
        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        processed = [norm(p) for p in patches]
        return torch.stack(processed)
